@echo off
rem Experimental build with clang-cl (LLVM): SAME flags as build.bat,
rem only the compiler changes. Separate output nf_clang.exe — the
rem official build stays MSVC (build.bat). vcvars64 is still needed:
rem clang-cl uses MSVC's and the Windows SDK's headers/libs.
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

echo [2/2] nf_clang.exe...
"C:\Program Files\LLVM\bin\clang-cl.exe" /std:c11 /O2 /W4 /arch:AVX2 /openmp /D_CRT_SECURE_NO_WARNINGS /I. nf.c nf_gguf.c nf_tokenizer.c nf_model.c /Fe:nf_clang.exe /link ws2_32.lib zstd\zstd_dec.lib "/LIBPATH:C:\Program Files\LLVM\lib" libomp.lib
if errorlevel 1 exit /b 1
echo OK: nf_clang.exe ready
