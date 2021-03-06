/* This file is an image processing operation for GEGL
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * GEGL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GEGL; if not, see <http://www.gnu.org/licenses/>.
 *
 * Copyright 2006 Øyvind Kolås <pippin@gimp.org>
 */

#include "config.h"
#include <glib/gi18n-lib.h>
#include <stdlib.h>

#ifdef GEGL_CHANT_PROPERTIES

gegl_chant_int_ui   (x, _("Width"),
                     1, G_MAXINT, 16, 1, 256, 1.5,
                     _("Horizontal width of cells pixels"))

gegl_chant_int_ui   (y, _("Height"),
                     1, G_MAXINT, 16, 1, 256, 1.5,
                     _("Vertical width of cells in pixels"))

gegl_chant_int_ui   (x_offset, _("X offset"),
                     -G_MAXINT, G_MAXINT, 0, -10, 10, 1.0,
                     _("Horizontal offset (from origin) for start of grid"))

gegl_chant_int_ui   (y_offset, _("Y offset"),
                     -G_MAXINT, G_MAXINT,  0, -10, 10, 1.0,
                     _("Vertical offset (from origin) for start of grid"))

gegl_chant_color    (color1, _("Color"),
                     "black",
                     _("One of the cell colors (defaults to 'black')"))

gegl_chant_color    (color2, _("Other color"),
                     "white",
                     _("The other cell color (defaults to 'white')"))

gegl_chant_format   (format, _("Babl Format"),
                     _("The babl format of the output"))

#else

#define GEGL_CHANT_TYPE_POINT_RENDER
#define GEGL_CHANT_C_FILE "checkerboard.c"

#include "gegl-chant.h"
#include <gegl-buffer-cl-iterator.h>
#include <gegl-debug.h>

static void
prepare (GeglOperation *operation)
{
  GeglChantO *o = GEGL_CHANT_PROPERTIES (operation);

  if (o->format)
    gegl_operation_set_format (operation, "output", o->format);
  else
    gegl_operation_set_format (operation, "output", babl_format ("RGBA float"));
}

static GeglRectangle
get_bounding_box (GeglOperation *operation)
{
  return gegl_rectangle_infinite_plane ();
}

static const char* checkerboard_cl_source =
"inline int tile_index (int coordinate, int stride)          \n"
"{                                                           \n"
"  if (coordinate >= 0)                                      \n"
"    return coordinate / stride;                             \n"
"  else                                                      \n"
"    return ((coordinate + 1) / stride) - 1;                 \n"
"}                                                           \n"
"                                                            \n"
"__kernel void kernel_checkerboard (__global float4 *out,    \n"
"                                   float4 color1,           \n"
"                                   float4 color2,           \n"
"                                   int square_width,        \n"
"                                   int square_height,       \n"
"                                   int x_offset,            \n"
"                                   int y_offset,            \n"
"                                   int roi_x,               \n"
"                                   int roi_y,               \n"
"                                   int roi_width)           \n"
"{                                                           \n"
"    int gidx = 0;                                           \n"
"    int gidy = get_global_id(0);                            \n"
"    float4 cur_color;                                       \n"
"    bool in_color1;                                         \n"
"                                                            \n"
"    int x = roi_x + gidx - x_offset;                        \n"
"    int y = roi_y + gidy - y_offset;                        \n"
"                                                            \n"
"    int tilex = tile_index (x, square_width);               \n"
"    int tiley = tile_index (y, square_height);              \n"
"                                                            \n"
"    if ((tilex + tiley) % 2 == 0)                           \n"
"      {                                                     \n"
"        cur_color = color1;                                 \n"
"        in_color1 = true;                                   \n"
"      }                                                     \n"
"    else                                                    \n"
"      {                                                     \n"
"        cur_color = color2;                                 \n"
"        in_color1 = false;                                  \n"
"      }                                                     \n"
"                                                            \n"
"    int stripe_end = (tilex + 1) * square_width;            \n"
"    int stripe_width = stripe_end - x;                      \n"
"    int gidx_max = roi_width;                               \n"
"                                                            \n"
"    while (gidx < gidx_max)                                 \n"
"      {                                                     \n"
"        out[gidx++ + gidy * roi_width] = cur_color;         \n"
"        stripe_width--;                                     \n"
"                                                            \n"
"        if (stripe_width == 0)                              \n"
"          {                                                 \n"
"            stripe_width = square_width;                    \n"
"                                                            \n"
"            if (in_color1)                                  \n"
"              cur_color = color2;                           \n"
"            else                                            \n"
"              cur_color = color1;                           \n"
"            in_color1 = !in_color1;                         \n"
"          }                                                 \n"
"      }                                                     \n"
"}                                                           \n";

#define TILE_INDEX(coordinate,stride) \
  (((coordinate) >= 0)?\
      (coordinate) / (stride):\
      ((((coordinate) + 1) /(stride)) - 1))


static GeglClRunData *cl_data = NULL;

static gboolean
checkerboard_cl_process (GeglOperation       *operation,
                         cl_mem               out_tex,
                         size_t               global_worksize,
                         const GeglRectangle *roi,
                         gint                 level)
{
  GeglChantO   *o           = GEGL_CHANT_PROPERTIES (operation);
  const Babl   *out_format  = gegl_operation_get_format (operation, "output");
  const size_t  gbl_size[1] = {roi->height};
  cl_int        cl_err      = 0;
  float         color1[4];
  float         color2[4];

  if (!cl_data)
  {
    const char *kernel_name[] = {"kernel_checkerboard", NULL};
    cl_data = gegl_cl_compile_and_build (checkerboard_cl_source, kernel_name);

    if (!cl_data)
      return TRUE;
  }

  gegl_color_get_pixel (o->color1, out_format, color1);
  gegl_color_get_pixel (o->color2, out_format, color2);

  cl_err = gegl_cl_set_kernel_args (cl_data->kernel[0],
                                    sizeof(cl_mem), &out_tex,
                                    sizeof(color1), &color1,
                                    sizeof(color2), &color2,
                                    sizeof(cl_int), &o->x,
                                    sizeof(cl_int), &o->y,
                                    sizeof(cl_int), &o->x_offset,
                                    sizeof(cl_int), &o->y_offset,
                                    sizeof(cl_int), &roi->x,
                                    sizeof(cl_int), &roi->y,
                                    sizeof(cl_int), &roi->width,
                                    NULL);
  CL_CHECK;

  cl_err = gegl_clEnqueueNDRangeKernel (gegl_cl_get_command_queue (),
                                        cl_data->kernel[0], 1,
                                        NULL, gbl_size, NULL,
                                        0, NULL, NULL);
  CL_CHECK;

  return FALSE;
error:
  return TRUE;
}

static gboolean
checkerboard_process (GeglOperation       *operation,
                      void                *out_buf,
                      glong                n_pixels,
                      const GeglRectangle *roi,
                      gint                 level)
{
  GeglChantO *o = GEGL_CHANT_PROPERTIES (operation);
  const Babl *out_format = gegl_operation_get_format (operation, "output");
  gint        pixel_size = babl_format_get_bytes_per_pixel (out_format);
  guchar     *out_pixel = out_buf;
  void       *color1 = alloca(pixel_size);
  void       *color2 = alloca(pixel_size);
  gint        y;
  const gint  x_min = roi->x - o->x_offset;
  const gint  y_min = roi->y - o->y_offset;
  const gint  x_max = roi->x + roi->width - o->x_offset;
  const gint  y_max = roi->y + roi->height - o->y_offset;

  const gint  square_width  = o->x;
  const gint  square_height = o->y;

  gegl_color_get_pixel (o->color1, out_format, color1);
  gegl_color_get_pixel (o->color2, out_format, color2);

  for (y = y_min; y < y_max; y++)
    {
      gint  x = x_min;
      void *cur_color;

      /* Figure out which box we're in */
      gint tilex = TILE_INDEX (x, square_width);
      gint tiley = TILE_INDEX (y, square_height);
      if ((tilex + tiley) % 2 == 0)
        cur_color = color1;
      else
        cur_color = color2;

      while (x < x_max)
        {
          /* Figure out how long this stripe is */
          gint count;
          gint stripe_end = (TILE_INDEX (x, square_width) + 1) * square_width;
               stripe_end = stripe_end > x_max ? x_max : stripe_end;

          count = stripe_end - x;

          gegl_memset_pattern (out_pixel, cur_color, pixel_size, count);
          out_pixel += count * pixel_size;
          x = stripe_end;

          if (cur_color == color1)
            cur_color = color2;
          else
            cur_color = color1;
        }
    }

  return TRUE;
}


static gboolean
operation_source_process (GeglOperation       *operation,
                          GeglBuffer          *output,
                          const GeglRectangle *result,
                          gint                 level)
{
  const Babl *out_format = gegl_operation_get_format (operation, "output");

  if ((result->width > 0) && (result->height > 0))
    {
      GeglBufferIterator *iter;
      if (gegl_operation_use_opencl (operation) &&
          babl_format_get_n_components (out_format) == 4 &&
          babl_format_get_type (out_format, 0) == babl_type ("float"))
        {
          GeglBufferClIterator *cl_iter;
          gboolean err;

          GEGL_NOTE (GEGL_DEBUG_OPENCL, "GEGL_OPERATION_POINT_RENDER: %s", GEGL_OPERATION_GET_CLASS (operation)->name);

          cl_iter = gegl_buffer_cl_iterator_new (output, result, out_format, GEGL_CL_BUFFER_WRITE);

          while (gegl_buffer_cl_iterator_next (cl_iter, &err) && !err)
            {
              err = checkerboard_cl_process (operation, cl_iter->tex[0], cl_iter->size[0], &cl_iter->roi[0], level);

              if (err)
                {
                  gegl_buffer_cl_iterator_stop (cl_iter);
                  break;
                }
            }

          if (err)
            GEGL_NOTE (GEGL_DEBUG_OPENCL, "Error: %s", GEGL_OPERATION_GET_CLASS (operation)->name);
          else
            return TRUE;
        }

      iter = gegl_buffer_iterator_new (output, result, level, out_format, GEGL_BUFFER_WRITE, GEGL_ABYSS_NONE);

      while (gegl_buffer_iterator_next (iter))
          checkerboard_process (operation, iter->data[0], iter->length, &iter->roi[0], level);
    }
  return TRUE;
}

static void
gegl_chant_class_init (GeglChantClass *klass)
{
  GeglOperationClass            *operation_class;
  GeglOperationSourceClass      *source_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  source_class = GEGL_OPERATION_SOURCE_CLASS (klass);

  source_class->process = operation_source_process;
  operation_class->get_bounding_box = get_bounding_box;
  operation_class->prepare = prepare;

  gegl_operation_class_set_keys (operation_class,
    "name",        "gegl:checkerboard",
    "categories",  "render",
    "description", _("Create a checkerboard pattern"),
    NULL);
}

#endif
