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


#ifdef GEGL_CHANT_PROPERTIES

gegl_chant_object (buffer, _("Buffer location"),
                   _("Write to an existing GeglBuffer"))

#else

#define GEGL_CHANT_TYPE_SINK
#define GEGL_CHANT_C_FILE "write-buffer.c"

#include "gegl-chant.h"

#include "gegl/gegl-debug.h"
#include "opencl/gegl-cl.h"
#include "gegl-buffer-cl-iterator.h"

static gboolean
process (GeglOperation       *operation,
         GeglBuffer          *input,
         const GeglRectangle *result,
         gint                 level)
{
  GeglChantO *o = GEGL_CHANT_PROPERTIES (operation);

  if (o->buffer)
    {
      GeglBuffer *output = GEGL_BUFFER (o->buffer);
      const Babl *in_format = gegl_buffer_get_format (input);
      const Babl *out_format = gegl_buffer_get_format (output);

      if (gegl_operation_use_opencl (operation)
          && gegl_cl_color_supported (in_format, out_format) == GEGL_CL_COLOR_CONVERT)
        {
          size_t size;
          gboolean err;
          cl_int cl_err = 0;

          GeglBufferClIterator *i = gegl_buffer_cl_iterator_new (output,
                                                                 result,
                                                                 out_format,
                                                                 GEGL_CL_BUFFER_WRITE);

          gint read = gegl_buffer_cl_iterator_add (i,
                                                   input,
                                                   result,
                                                   out_format,
                                                   GEGL_CL_BUFFER_READ,
                                                   GEGL_ABYSS_NONE);

          gegl_cl_color_babl (out_format, &size);

          GEGL_NOTE (GEGL_DEBUG_OPENCL,
                     "write-buffer: "
                     "%p %p %s %s {%d %d %d %d}",
                     input,
                     output,
                     babl_get_name (in_format),
                     babl_get_name (out_format),
                     result->x,
                     result->y,
                     result->width,
                     result->height);

          while (gegl_buffer_cl_iterator_next (i, &err))
            {
              if (err) break;

              cl_err = gegl_clEnqueueCopyBuffer (gegl_cl_get_command_queue (),
                                                 i->tex[read],
                                                 i->tex[0],
                                                 0,
                                                 0,
                                                 i->size[0] * size,
                                                 0,
                                                 NULL,
                                                 NULL);

              if (cl_err != CL_SUCCESS)
                {
                  GEGL_NOTE (GEGL_DEBUG_OPENCL, "Error: %s", gegl_cl_errstring (cl_err));
                  break;
                }
            }

          if (cl_err || err)
            gegl_buffer_copy (input, result, output, result);
        }
      else
        gegl_buffer_copy (input, result, output, result);

      gegl_buffer_flush (output);
    }

  return TRUE;
}

static void
dispose (GObject *object)
{
  GeglChantO *o = GEGL_CHANT_PROPERTIES (object);

  if (o->buffer)
    {
      g_object_unref (o->buffer);
      o->buffer = NULL;
    }

  G_OBJECT_CLASS (gegl_chant_parent_class)->dispose (object);
}

static void
gegl_chant_class_init (GeglChantClass *klass)
{
  GeglOperationClass     *operation_class;
  GeglOperationSinkClass *sink_class;

  operation_class = GEGL_OPERATION_CLASS (klass);
  sink_class      = GEGL_OPERATION_SINK_CLASS (klass);

  sink_class->process = process;
  sink_class->needs_full = FALSE;

  G_OBJECT_CLASS (klass)->dispose = dispose;

  gegl_operation_class_set_keys (operation_class,
    "name"       , "gegl:write-buffer",
    "categories" , "programming:output",
    "description", _("A GEGL buffer destination surface."),
    NULL);
}

#endif
