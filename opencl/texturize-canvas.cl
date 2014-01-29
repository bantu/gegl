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
 * Copyright (C) 2014 Andreas Fischer (andreas.fischer@student.kit.edu)
 */

__kernel void cl_texturize_canvas(__global const float *src,
                                  __global       float *dest,
                                  __global const float *texture,
                                           const float mult,
                                           const int   xm,
                                           const int   ym,
                                           const int   offs,
                                           const int   components,
                                           const int   has_alpha,
                                           const int   roi_x,
                                           const int   roi_y,
                                           const int   roi_width)
{
  int buffer_index = get_global_id(0);
  int color_index = buffer_index % (components + has_alpha);
  if (color_index < components)
    {
      int col_index = (buffer_index / (components + has_alpha)) % roi_width;
      int row_index = (buffer_index / (components + has_alpha)) / roi_width;
      int texture_index = ((roi_x + col_index) & 127) * xm +
                          ((roi_y + row_index) & 127) * ym +
                          offs;
      float color = mult * texture [texture_index] + src [buffer_index];
      dest [buffer_index] = clamp (color, 0.0f, 1.0f);
    }
  else
    {
      // Copy alpha channel
      dest [buffer_index] = src [buffer_index];
    }
}
