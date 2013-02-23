#ifndef SIMD_H
#define SIMD_H

#include <snlsys/snlsys.h>

#if defined(SIMD_SHARED_BUILD)
  #define SIMD_API EXPORT_SYM
#else
  #define SIMD_API IMPORT_SYM
#endif

#ifdef SIMD_SSE2
  #include "sse/sse.h"
#else
  #error unsupported_platform
#endif

#endif /* SIMD_H */

