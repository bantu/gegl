static const char *texturize_canvas_cl_source =
"#define CLAMP(val,lo,hi) (val < lo) ? lo : ((hi < val) ? hi : val )        \n"
"__kernel void cl_texturize_canvas(__global const float * in,               \n"
"                                  __global float * out,                    \n"
"                                  __global const float * sdata,            \n"
"                                  const int x,                             \n"
"                                  const int y,                             \n"
"                                  const int xm,                            \n"
"                                  const int ym,                            \n"
"                                  const int offs,                          \n"
"                                  const float mult,                        \n"
"                                  const int components,                    \n"
"                                  const int has_alpha)                     \n"
"{                                                                          \n"
"    int col = get_global_id(0);                                            \n"
"    int row = get_global_id(1);                                            \n"
"    int step = components + has_alpha;                                     \n"
"    int index = step * (row * get_global_size(0) + col);                   \n"
"    int canvas_index = ((x + col) & 127) * xm +                            \n"
"                       ((y + row) & 127) * ym + offs;                      \n"
"    float color;                                                           \n"
"    int i;                                                                 \n"
"    float tmp = mult * sdata[canvas_index];                                \n"
"    for(i=0; i<components; ++i)                                            \n"
"    {                                                                      \n"
"       color = tmp + in[index];                                            \n"
"       out[index++] = CLAMP(color,0.0f,1.0f);                              \n"
"    }                                                                      \n"
"    if(has_alpha)                                                          \n"
"       out[index] = in[index];                                             \n"
"}                                                                          \n"
;
