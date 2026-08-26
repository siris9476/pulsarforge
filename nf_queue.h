/* nf_queue.h — generic thread-safe queue (fixed-capacity ring buffer),
 * extracted from nf.c because nf_model.c needs it too
 * (Phase 1 async prefetch in moe_ffn_xsess). Verbatim move, zero logic
 * change — validated standalone with `nf debugqueue` before and after.
 *
 * Isolated primitive: native Win32 CRITICAL_SECTION+CONDITION_VARIABLE,
 * no external dependency. Fixed-size items (item_size bytes each),
 * copied by value — no shared pointers to manage. Blocking push/pop;
 * nf_queue_close() unblocks every pending wait and makes every
 * subsequent pop on an empty queue fail (never a silent deadlock at
 * shutdown). ALWAYS `while(cond) Sleep...`, never `if`: Win32's
 * CONDITION_VARIABLEs allow spurious wakeups. */
#ifndef NF_QUEUE_H
#define NF_QUEUE_H

#ifdef _WIN32
#include <windows.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char *buf;
    size_t item_size;
    int capacity;
    int head, tail, count;
    int closed;
    CRITICAL_SECTION lock;
    CONDITION_VARIABLE not_empty;
    CONDITION_VARIABLE not_full;
} nf_queue;

static void nf_queue_init(nf_queue *q, size_t item_size, int capacity) {
    q->buf = malloc(item_size * (size_t)capacity);
    q->item_size = item_size;
    q->capacity = capacity;
    q->head = q->tail = q->count = 0;
    q->closed = 0;
    InitializeCriticalSection(&q->lock);
    InitializeConditionVariable(&q->not_empty);
    InitializeConditionVariable(&q->not_full);
}

static void nf_queue_free(nf_queue *q) {
    DeleteCriticalSection(&q->lock);
    free(q->buf);
}

/* Unblocks every pending push/pop: used at shutdown. After closing,
 * push no longer inserts anything (returns without blocking), pop
 * returns -1 as soon as the queue drains (never blocks forever). */
static void nf_queue_close(nf_queue *q) {
    EnterCriticalSection(&q->lock);
    q->closed = 1;
    LeaveCriticalSection(&q->lock);
    WakeAllConditionVariable(&q->not_empty);
    WakeAllConditionVariable(&q->not_full);
}

/* Returns 0 if inserted, -1 if the queue is closed (the item doesn't go in). */
static int nf_queue_push(nf_queue *q, const void *item) {
    EnterCriticalSection(&q->lock);
    while (q->count == q->capacity && !q->closed)
        SleepConditionVariableCS(&q->not_full, &q->lock, INFINITE);
    if (q->closed) { LeaveCriticalSection(&q->lock); return -1; }
    memcpy(q->buf + (size_t)q->tail * q->item_size, item, q->item_size);
    q->tail = (q->tail + 1) % q->capacity;
    q->count++;
    LeaveCriticalSection(&q->lock);
    WakeConditionVariable(&q->not_empty);
    return 0;
}

/* Returns 0 if an item was extracted into out, -1 if the queue is
 * closed AND empty (no other item will ever arrive). */
static int nf_queue_pop(nf_queue *q, void *out) {
    EnterCriticalSection(&q->lock);
    while (q->count == 0 && !q->closed)
        SleepConditionVariableCS(&q->not_empty, &q->lock, INFINITE);
    if (q->count == 0) { LeaveCriticalSection(&q->lock); return -1; }
    memcpy(out, q->buf + (size_t)q->head * q->item_size, q->item_size);
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    LeaveCriticalSection(&q->lock);
    WakeConditionVariable(&q->not_full);
    return 0;
}

/* Demand-first priority queue:
 * inserts at the HEAD of the ring instead of the tail — the next pop
 * extracts THIS item before the whole backlog already queued. Used by
 * the MANDATORY fetches of glm-streaming to overtake pending
 * speculative prefetches (a standard pattern: demand reads jump ahead
 * of readahead — Linux page cache, I/O scheduler, ProMoE ASPLOS'25).
 * Same lock, same condition variables, same single consumer: the
 * PILOT lessons hold by construction — only WHERE in the ring the item
 * gets written changes. An item already in flight on a worker cannot
 * be overtaken (no preempting a ReadFile): the priority applies only
 * to what's still queued. */
static int nf_queue_push_front(nf_queue *q, const void *item) {
    EnterCriticalSection(&q->lock);
    while (q->count == q->capacity && !q->closed)
        SleepConditionVariableCS(&q->not_full, &q->lock, INFINITE);
    if (q->closed) { LeaveCriticalSection(&q->lock); return -1; }
    q->head = (q->head - 1 + q->capacity) % q->capacity;
    memcpy(q->buf + (size_t)q->head * q->item_size, item, q->item_size);
    q->count++;
    LeaveCriticalSection(&q->lock);
    WakeConditionVariable(&q->not_empty);
    return 0;
}

/* M8 Phase 3b: NON-blocking variant — returns immediately with -1 if
 * the queue is empty (open or closed doesn't matter), never waits.
 * Used by the scheduler to drain incoming requests between ticks
 * without stalling when there aren't any. Same already-validated data
 * structure (nf debugqueue): this variant only removes the wait on
 * the condition variable, the rest is identical. */
static int nf_queue_try_pop(nf_queue *q, void *out) {
    EnterCriticalSection(&q->lock);
    if (q->count == 0) { LeaveCriticalSection(&q->lock); return -1; }
    memcpy(out, q->buf + (size_t)q->head * q->item_size, q->item_size);
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    LeaveCriticalSection(&q->lock);
    WakeConditionVariable(&q->not_full);
    return 0;
}

/* Phase 2 PILOT: like nf_queue_pop but with a millisecond
 * cap on the wait — for a best-effort wait that must NEVER be able to
 * block forever (unlike nf_queue_pop, meant for a reliable
 * producer/consumer where an indefinite block is acceptable). Returns
 * 0 if extracted, -1 if the queue is closed AND empty (no other item
 * will ever arrive), -2 if the cap expired with nothing arriving
 * (queue still open — it might arrive later, the caller decides
 * whether to give up). */
static int nf_queue_pop_timeout(nf_queue *q, void *out, DWORD timeout_ms) {
    EnterCriticalSection(&q->lock);
    while (q->count == 0 && !q->closed) {
        if (!SleepConditionVariableCS(&q->not_empty, &q->lock, timeout_ms)) {
            LeaveCriticalSection(&q->lock);
            return -2;   /* timeout (or a rare error treated as one) */
        }
    }
    if (q->count == 0) { LeaveCriticalSection(&q->lock); return -1; }
    memcpy(out, q->buf + (size_t)q->head * q->item_size, q->item_size);
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    LeaveCriticalSection(&q->lock);
    WakeConditionVariable(&q->not_full);
    return 0;
}

#else
/* ------------------------------------------------------------ POSIX
 * (Linux port project): same API, same semantics —
 * blocking push/pop, close that drains and never deadlocks, ALWAYS a
 * while loop around waits (pthread also has spurious wakeups). */
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    unsigned char *buf;
    size_t item_size;
    int capacity;
    int head, tail, count;
    int closed;
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} nf_queue;

static void nf_queue_init(nf_queue *q, size_t item_size, int capacity) {
    q->buf = malloc(item_size * (size_t)capacity);
    q->item_size = item_size;
    q->capacity = capacity;
    q->head = q->tail = q->count = 0;
    q->closed = 0;
    pthread_mutex_init(&q->lock, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

static void nf_queue_free(nf_queue *q) {
    pthread_mutex_destroy(&q->lock);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
    free(q->buf);
}

static void nf_queue_close(nf_queue *q) {
    pthread_mutex_lock(&q->lock);
    q->closed = 1;
    pthread_mutex_unlock(&q->lock);
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
}

static int nf_queue_push(nf_queue *q, const void *item) {
    pthread_mutex_lock(&q->lock);
    while (q->count == q->capacity && !q->closed)
        pthread_cond_wait(&q->not_full, &q->lock);
    if (q->closed) { pthread_mutex_unlock(&q->lock); return -1; }
    memcpy(q->buf + (size_t)q->tail * q->item_size, item, q->item_size);
    q->tail = (q->tail + 1) % q->capacity;
    q->count++;
    pthread_mutex_unlock(&q->lock);
    pthread_cond_signal(&q->not_empty);
    return 0;
}

static int nf_queue_pop(nf_queue *q, void *out) {
    pthread_mutex_lock(&q->lock);
    while (q->count == 0 && !q->closed)
        pthread_cond_wait(&q->not_empty, &q->lock);
    if (q->count == 0) { pthread_mutex_unlock(&q->lock); return -1; }
    memcpy(out, q->buf + (size_t)q->head * q->item_size, q->item_size);
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    pthread_mutex_unlock(&q->lock);
    pthread_cond_signal(&q->not_full);
    return 0;
}

static int nf_queue_push_front(nf_queue *q, const void *item) {
    pthread_mutex_lock(&q->lock);
    while (q->count == q->capacity && !q->closed)
        pthread_cond_wait(&q->not_full, &q->lock);
    if (q->closed) { pthread_mutex_unlock(&q->lock); return -1; }
    q->head = (q->head - 1 + q->capacity) % q->capacity;
    memcpy(q->buf + (size_t)q->head * q->item_size, item, q->item_size);
    q->count++;
    pthread_mutex_unlock(&q->lock);
    pthread_cond_signal(&q->not_empty);
    return 0;
}

static int nf_queue_try_pop(nf_queue *q, void *out) {
    pthread_mutex_lock(&q->lock);
    if (q->count == 0) { pthread_mutex_unlock(&q->lock); return -1; }
    memcpy(out, q->buf + (size_t)q->head * q->item_size, q->item_size);
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    pthread_mutex_unlock(&q->lock);
    pthread_cond_signal(&q->not_full);
    return 0;
}

static int nf_queue_pop_timeout(nf_queue *q, void *out, unsigned timeout_ms) {
    struct timespec abst;
    clock_gettime(CLOCK_REALTIME, &abst);
    abst.tv_sec += timeout_ms / 1000;
    abst.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
    if (abst.tv_nsec >= 1000000000L) { abst.tv_sec++; abst.tv_nsec -= 1000000000L; }
    pthread_mutex_lock(&q->lock);
    while (q->count == 0 && !q->closed) {
        if (pthread_cond_timedwait(&q->not_empty, &q->lock, &abst) == ETIMEDOUT) {
            pthread_mutex_unlock(&q->lock);
            return -2;
        }
    }
    if (q->count == 0) { pthread_mutex_unlock(&q->lock); return -1; }
    memcpy(out, q->buf + (size_t)q->head * q->item_size, q->item_size);
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    pthread_mutex_unlock(&q->lock);
    pthread_cond_signal(&q->not_full);
    return 0;
}

#endif /* _WIN32 */
#endif /* NF_QUEUE_H */
