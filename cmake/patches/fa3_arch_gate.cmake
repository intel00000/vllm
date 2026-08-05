# Gate the FA3 Hopper instantiation farm on an sm_90 target actually being
# present. Upstream flash-attention (caaa4eb59) compiles all ~250
# hopper/instantiations TUs whenever CUDA >= 12.0, even when FA3_ARCHS ends up
# empty (pure sm_80 / sm_120 builds) -- the objects then carry no usable device
# code (nvcc default arch) and cost ~20-30 min of build time.
# Applied by the FetchContent PATCH_COMMAND in
# cmake/external_projects/vllm_flash_attn.cmake; runs in the populated source
# dir; idempotent via the hb-fa3-arch-gate marker.
file(READ CMakeLists.txt _hb_content)
if(_hb_content MATCHES "hb-fa3-arch-gate")
  return()
endif()
string(REPLACE
  "set(FA3_ENABLED ON)"
  "set(FA3_ENABLED ON) # hb-fa3-arch-gate\nif(DEFINED CUDA_ARCHS AND NOT \"\${CUDA_ARCHS}\" MATCHES \"9[.]0\")\n  set(FA3_ENABLED OFF)\nendif()"
  _hb_content "${_hb_content}")
file(WRITE CMakeLists.txt "${_hb_content}")
message(STATUS "hb: FA3 arch gate applied to vllm-flash-attn CMakeLists.txt")
