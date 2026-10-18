#ifndef P7_FIELD_H
#define P7_FIELD_H
// P7 reuses P6's scorer, sinks and geometry ABI unchanged; only the per-history
// capacity grows to the common +-20 cm layout (15 regions, 3,420 bins).
#include "p6_field.h"
#ifndef P5_GPU
#include <cstddef>
#endif
#ifndef P7_FAULT
#define P7_FAULT 0
#endif
enum {P7_REGIONS=15,P7_BINS=3420};
typedef P6Geometry P7Geometry;
static_assert(sizeof(P7Geometry)==1144,"P7 CPU/CUDA geometry ABI (same as P6)");
#ifndef P5_GPU
static_assert(offsetof(P6Geometry,hi)==360 && offsetof(P6Geometry,origins)==720 && offsetof(P6Geometry,widths)==840
              && offsetof(P6Geometry,axes)==960 && offsetof(P6Geometry,sizes)==1020 && offsetof(P6Geometry,offsets)==1080,
              "P7 geometry member offsets");
#endif
#endif
