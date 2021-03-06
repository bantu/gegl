/* This file is part of GEGL.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 * 2013 Daniel Sabo
 */

#ifndef __GEGL_ALGORITHMS_H__
#define __GEGL_ALGORITHMS_H__

G_BEGIN_DECLS

#define GEGL_SCALE_EPSILON 1.e-6

/* Attempt to resample with a 3x3 boxfilter, if no boxfilter is
 * available for #format fall back to nearest neighbor.
 * #scale is assumed to be between 0.5 and +inf.
 */
void gegl_resample_boxfilter (guchar              *dest_buf,
                              const guchar        *source_buf,
                              const GeglRectangle *dst_rect,
                              const GeglRectangle *src_rect,
                              gint                 s_rowstride,
                              gdouble              scale,
                              const Babl          *format,
                              gint                 d_rowstride);

void gegl_resample_boxfilter_float (guchar              *dest_buf,
                                    const guchar        *source_buf,
                                    const GeglRectangle *dst_rect,
                                    const GeglRectangle *src_rect,
                                    gint                 s_rowstride,
                                    gdouble              scale,
                                    gint                 components,
                                    gint                 d_rowstride);

void gegl_resample_boxfilter_u32 (guchar              *dest_buf,
                                  const guchar        *source_buf,
                                  const GeglRectangle *dst_rect,
                                  const GeglRectangle *src_rect,
                                  gint                 s_rowstride,
                                  gdouble              scale,
                                  gint                 components,
                                  gint                 d_rowstride);

void gegl_resample_boxfilter_u16 (guchar              *dest_buf,
                                  const guchar        *source_buf,
                                  const GeglRectangle *dst_rect,
                                  const GeglRectangle *src_rect,
                                  gint                 s_rowstride,
                                  gdouble              scale,
                                  gint                 components,
                                  gint                 d_rowstride);

void gegl_resample_boxfilter_u8 (guchar              *dest_buf,
                                 const guchar        *source_buf,
                                 const GeglRectangle *dst_rect,
                                 const GeglRectangle *src_rect,
                                 gint                 s_rowstride,
                                 gdouble              scale,
                                 gint                 components,
                                 gint                 d_rowstride);

void gegl_resample_nearest (guchar              *dst,
                            const guchar        *src,
                            const GeglRectangle *dst_rect,
                            const GeglRectangle *src_rect,
                            gint                 src_stride,
                            gdouble              scale,
                            gint                 bpp,
                            gint                 dst_stride);

G_END_DECLS

#endif /* __GEGL_ALGORITHMS_H__ */