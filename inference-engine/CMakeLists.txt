cmake_minimum_required(VERSION 3.21)
project(xenon_inference_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This project links against a *pre-built* rwkv.cpp (see ../rwkv.cpp), rather than
# add_subdirectory()-ing it, so that the same source tree here can produce two independent
# apps -- CPU-only and CUDA-enabled -- by pointing at two different rwkv.cpp CMake build
# directories (rwkv.cpp/build-cpu and rwkv.cpp/build-cuda), without recompiling ggml/rwkv
# twice inside this project. See inference-engine/README.md for exact build commands.
set(XENON_RWKV_BUILD_DIR "" CACHE PATH "Path to a built rwkv.cpp CMake build dir (contains bin/Release/rwkv.dll)")
option(XENON_HAS_CUDA "Set ON when XENON_RWKV_BUILD_DIR was configured with -DRWKV_CUBLAS=ON" OFF)

if (NOT XENON_RWKV_BUILD_DIR)
    message(FATAL_ERROR "Set -DXENON_RWKV_BUILD_DIR=<path to rwkv.cpp/build-cpu or rwkv.cpp/build-cuda>")
endif()

# Windows multi-config (Visual Studio) puts rwkv.dll under bin/Release and the import lib
# under Release/. Ninja/Makefiles on Linux put librwkv.so in the build directory itself.
set(_rwkv_shared_candidates
    "${XENON_RWKV_BUILD_DIR}/bin/Release/rwkv.dll"
    "${XENON_RWKV_BUILD_DIR}/bin/rwkv.dll"
    "${XENON_RWKV_BUILD_DIR}/librwkv.so"
    "${XENON_RWKV_BUILD_DIR}/lib/librwkv.so"
    "${XENON_RWKV_BUILD_DIR}/Release/librwkv.so"
)
set(_rwkv_shared "")
foreach(_candidate ${_rwkv_shared_candidates})
    if (EXISTS "${_candidate}")
        set(_rwkv_shared "${_candidate}")
        break()
    endif()
endforeach()
if (NOT _rwkv_shared)
    message(FATAL_ERROR
        "No rwkv shared library under ${XENON_RWKV_BUILD_DIR}. Looked for rwkv.dll and librwkv.so.")
endif()

add_library(rwkv_imported SHARED IMPORTED)
set_target_properties(rwkv_imported PROPERTIES IMPORTED_LOCATION "${_rwkv_shared}")
if (WIN32)
    set(_rwkv_implib_candidates
        "${XENON_RWKV_BUILD_DIR}/Release/rwkv.lib"
        "${XENON_RWKV_BUILD_DIR}/rwkv.lib"
    )
    set(_rwkv_implib "")
    foreach(_candidate ${_rwkv_implib_candidates})
        if (EXISTS "${_candidate}")
            set(_rwkv_implib "${_candidate}")
            break()
        endif()
    endforeach()
    if (NOT _rwkv_implib)
        message(FATAL_ERROR "rwkv.lib not found under ${XENON_RWKV_BUILD_DIR}")
    endif()
    set_target_properties(rwkv_imported PROPERTIES IMPORTED_IMPLIB "${_rwkv_implib}")
endif()
target_include_directories(rwkv_imported INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/rwkv.cpp")

add_library(xenon_inference SHARED
    src/xenon_inference.cpp
    src/world_tokenizer.cpp
)
target_include_directories(xenon_inference PUBLIC src)
target_link_libraries(xenon_inference PUBLIC rwkv_imported)
target_compile_definitions(xenon_inference
    PRIVATE XENON_INFERENCE_BUILD XENON_HAS_CUDA=$<BOOL:${XENON_HAS_CUDA}>
    PUBLIC XENON_INFERENCE_SHARED
)

add_executable(test_inference src/test_inference.cpp)
target_link_libraries(test_inference PRIVATE xenon_inference)
if (WIN32)
    target_link_libraries(test_inference PRIVATE psapi)
endif()

# Phase 8: correctness gate + measurement harness for the incremental state API (xenon_state_*,
# xenon_prefill, xenon_generate_with_state) -- see the file's own top comment.
add_executable(test_state_cache src/test_state_cache.cpp)
target_link_libraries(test_state_cache PRIVATE xenon_inference)
if (WIN32)
    target_link_libraries(test_state_cache PRIVATE psapi)
endif()

set_target_properties(xenon_inference test_inference test_state_cache PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
if (UNIX AND NOT APPLE)
    # The Tauri app and the CLI both load this .so by directory. $ORIGIN lets librwkv.so
    # resolve from beside libxenon_inference.so without a global ldconfig entry.
    set_target_properties(xenon_inference PROPERTIES
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
    )
endif()

# The rwkv shared library has to sit next to the test binary (and, on Linux, next to
# libxenon_inference.so) or the dynamic loader will not find it.
foreach(_xenon_test_target test_inference test_state_cache)
    add_custom_command(TARGET ${_xenon_test_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_rwkv_shared}"
            "$<TARGET_FILE_DIR:${_xenon_test_target}>/"
        COMMENT "Copying rwkv shared library next to ${_xenon_test_target}"
    )

    # Follow-up fix: world_vocab.bin (the World tokenizer's vocab file) must ship next to the
    # test .exe so the build output folder is self-contained and it can be run directly from
    # build-cpu-app/bin/Release or build-cuda-app/bin/Release without depending on the current
    # working directory being the project root. Previously this was a manual copy step nobody
    # remembered to run; both harnesses' default --vocab now also resolves relative to the
    # executable's own directory (see get_exe_dir() in each), so this copy is what makes that
    # default actually resolve to a real file.
    add_custom_command(TARGET ${_xenon_test_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/data/world_vocab.bin"
            "$<TARGET_FILE_DIR:${_xenon_test_target}>/world_vocab.bin"
        COMMENT "Copying world_vocab.bin next to ${_xenon_test_target}"
    )
endforeach()

if (UNIX AND NOT APPLE)
    add_custom_command(TARGET xenon_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_rwkv_shared}"
            "$<TARGET_FILE_DIR:xenon_inference>/"
        COMMENT "Copying librwkv.so next to libxenon_inference.so"
    )
endif()

# Extra runtime DLLs discovered as real, hard requirements while getting this to actually run
# (not just build) on this machine -- best-effort app-local deployment so test_inference.exe
# doesn't depend on PATH/registry state, which also matches this project's eventual
# run-off-a-USB-drive goal. Silently skipped if not found (e.g. different VS/CUDA install
# layout) rather than failing the build over it.
if (WIN32)
    set(_xenon_runtime_dlls "")

    if (XENON_HAS_CUDA)
        # CUDA 12+ toolkits on Windows ship cublas64_*.dll/cublasLt64_*.dll under bin/x64
        # rather than directly under bin/ (layout changed at some point in CUDA's history) --
        # rwkv.dll links these dynamically (only cudart is statically linked, per
        # rwkv.cpp/CMakeLists.txt), so without this the CUDA build fails to *run* with
        # "cublas64_13.dll: cannot open shared object file", even though it built fine.
        file(GLOB _xenon_cublas_dlls
            "$ENV{CUDA_PATH}/bin/cublas64_*.dll"
            "$ENV{CUDA_PATH}/bin/cublasLt64_*.dll"
            "$ENV{CUDA_PATH}/bin/x64/cublas64_*.dll"
            "$ENV{CUDA_PATH}/bin/x64/cublasLt64_*.dll"
        )
        list(APPEND _xenon_runtime_dlls ${_xenon_cublas_dlls})
    endif()

    # VC++ / Universal CRT redistributable DLLs. Discovered as a real requirement here: the
    # CUDA-linked rwkv.dll failed to load with "api-ms-win-crt-locale-l1-1-0.dll: cannot open
    # shared object file" on this machine, even though the CPU-only build ran fine without it.
    # Copying these app-local fixes it and is harmless for the CPU-only config too.
    file(GLOB _xenon_vc_redist_dirs
        "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT"
    )
    file(GLOB _xenon_ucrt_redist_dirs
        "C:/Program Files (x86)/Windows Kits/10/Redist/*/ucrt/DLLs/x64"
    )
    foreach(_dir ${_xenon_vc_redist_dirs} ${_xenon_ucrt_redist_dirs})
        file(GLOB _dlls "${_dir}/*.dll")
        list(APPEND _xenon_runtime_dlls ${_dlls})
    endforeach()

    foreach(_dll ${_xenon_runtime_dlls})
        add_custom_command(TARGET test_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:test_inference>/"
        )
    endforeach()
endif()
