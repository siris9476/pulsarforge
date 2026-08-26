@echo off
rem PulsarForge build (MSVC, any Visual Studio 2022 edition).
rem Builds the zstd lib (decoder only, for the .forgez/.forgezh
rem containers) if missing, then nf.exe. See build_posix.sh for
rem gcc/Linux (the Makefile is stale, not maintained).
setlocal
set VCV=
for %%E in (Enterprise Professional Community BuildTools) do (
    if not defined VCV if exist "C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" set "VCV=C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCV (
    echo ERROR: vcvars64.bat not found - requires Visual Studio 2022 with C++
    exit /b 1
)
call "%VCV%" >nul
cd /d "%~dp0"

if not exist zstd\zstd_dec.lib (
    echo [1/2] zstd library ^(decoder^)...
    pushd zstd
    cl /nologo /c /O2 /w /DZSTD_DISABLE_ASM /DZSTD_LEGACY_SUPPORT=0 /I. common\debug.c common\entropy_common.c common\error_private.c common\fse_decompress.c common\xxhash.c common\zstd_common.c decompress\huf_decompress.c decompress\zstd_ddict.c decompress\zstd_decompress.c decompress\zstd_decompress_block.c
    if errorlevel 1 ( popd & exit /b 1 )
    lib /nologo /out:zstd_dec.lib debug.obj entropy_common.obj error_private.obj fse_decompress.obj xxhash.obj zstd_common.obj huf_decompress.obj zstd_ddict.obj zstd_decompress.obj zstd_decompress_block.obj
    if errorlevel 1 ( popd & exit /b 1 )
    del *.obj
    popd
)

echo [2/2] nf.exe...
rem ws2_32: Winsock for `nf serve`
cl /nologo /std:c11 /O2 /W4 /arch:AVX2 /openmp /D_CRT_SECURE_NO_WARNINGS /I. nf.c nf_gguf.c nf_tokenizer.c nf_model.c ws2_32.lib zstd\zstd_dec.lib /Fe:nf.exe
if errorlevel 1 exit /b 1
echo OK: nf.exe ready
