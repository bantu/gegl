
### GObject introspection
-include $(INTROSPECTION_MAKEFILE)
INTROSPECTION_GIRS =
INTROSPECTION_SCANNER_ARGS = --warn-all --add-include-path="$(srcdir)" --add-include-path="$(shell $(PKG_CONFIG) --define-variable=datadir="$(datadir)" --variable=girdir gobject-introspection-1.0)"
INTROSPECTION_COMPILER_ARGS = --includedir="$(srcdir)" --includedir="$(shell $(PKG_CONFIG) --define-variable=datadir="$(datadir)" --variable=girdir gobject-introspection-1.0)"
INTROSPECTION_SCANNER_ENV = CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" CPPFLAGS="$(CPPFLAGS)" CXXFLAGS="$(CXXFLAGS)"

if PLATFORM_WIN32
no_undefined = -no-undefined
endif

SUBDIRS = buffer graph module operation process property-types opencl

CLEANFILES =

AM_CPPFLAGS = \
	-I$(top_srcdir) \
	-I$(top_builddir)/gegl \
	-I$(top_srcdir)/gegl \
	-I$(top_builddir)/gegl/buffer \
	-I$(top_srcdir)/gegl/buffer \
	-I$(top_builddir)/gegl/graph \
	-I$(top_srcdir)/gegl/graph \
	-I$(top_builddir)/gegl/operation \
	-I$(top_srcdir)/gegl/operation \
	-I$(top_builddir)/gegl/opencl \
	-I$(top_srcdir)/gegl/opencl \
	-I$(top_builddir)/gegl/process \
	-I$(top_srcdir)/gegl/process \
	-I$(top_builddir)/gegl/property-types \
	-I$(top_srcdir)/gegl/property-types \
	-DLIBDIR=\""$(libdir)"\" \
	-DGEGL_LOCALEDIR=\""$(GEGL_LOCALEDIR)"\"

AM_CFLAGS = $(DEP_CFLAGS) $(BABL_CFLAGS)

AM_LDFLAGS = \
	$(no_undefined) -export-dynamic -version-info $(GEGL_LIBRARY_VERSION)

LIBS = \
	$(DEP_LIBS) $(BABL_LIBS)

GEGL_publicdir = $(includedir)/gegl-$(GEGL_API_VERSION)

GEGL_introspectable_headers =	\
	gegl.h				\
	gegl-types.h			\
	gegl-apply.h			\
	gegl-enums.h			\
	gegl-operations-util.h		\
	gegl-utils.h			\
	gegl-matrix.h			\
	gegl-lookup.h			\
	gegl-random.h			\
	gegl-init.h			\
	gegl-version.h			\
	buffer/gegl-buffer.h		\
	buffer/gegl-buffer-iterator.h	\
	buffer/gegl-buffer-backend.h	\
	buffer/gegl-tile-backend.h		\
	buffer/gegl-tile-handler.h		\
	buffer/gegl-tile-source.h		\
	graph/gegl-node.h			\
	process/gegl-graph-debug.h		\
	process/gegl-processor.h		\
	property-types/gegl-paramspecs.h	\
	property-types/gegl-color.h		\
	property-types/gegl-path.h		\
	property-types/gegl-curve.h

GEGL_public_HEADERS = \
	$(GEGL_introspectable_headers)	\
	gegl-c.h			\
	gegl-chant.h			\
	gegl-cpuaccel.h			\
	gegl-plugin.h			\
	buffer/gegl-tile.h \
	buffer/gegl-buffer-cl-iterator.h

GEGL_sources = \
	gegl-c.c			\
	gegl-algorithms.c \
	gegl-apply.c			\
	gegl-config.c			\
	gegl-cpuaccel.c			\
	gegl-dot.c			\
	gegl-dot-visitor.c		\
	gegl-enums.c			\
	gegl-init.c			\
	gegl-instrument.c		\
	gegl-introspection-support.c	\
	gegl-utils.c			\
	gegl-lookup.c			\
	gegl-xml.c			\
	gegl-random.c			\
	gegl-matrix.c			\
	\
	gegl-algorithms.h \
	gegl-chant.h			\
	gegl-config.h			\
	gegl-cpuaccel.h			\
	gegl-cpuaccel-private.h		\
	gegl-debug.h			\
	gegl-dot.h			\
	gegl-dot-visitor.h		\
	gegl-init-private.h		\
	gegl-instrument.h		\
	gegl-introspection-support.h	\
	gegl-lookup.h			\
	gegl-matrix.h			\
	gegl-module.h			\
	gegl-plugin.h			\
	gegl-random-private.h		\
	gegl-types-internal.h		\
	gegl-xml.h

EXTRA_DIST = gegl-algorithms-boxfilter.inc

lib_LTLIBRARIES = libgegl-@GEGL_API_VERSION@.la

libgegl_@GEGL_API_VERSION@_la_SOURCES = $(GEGL_sources) $(GEGL_public_HEADERS)
libgegl_@GEGL_API_VERSION@_la_LIBADD = \
	$(top_builddir)/gegl/buffer/libbuffer.la		\
	$(top_builddir)/gegl/module/libmodule.la		\
	$(top_builddir)/gegl/operation/liboperation.la		\
	$(top_builddir)/gegl/graph/libgraph.la			\
	$(top_builddir)/gegl/process/libprocess.la		\
	$(top_builddir)/gegl/property-types/libpropertytypes.la \
	$(top_builddir)/gegl/opencl/libcl.la


if HAVE_INTROSPECTION
introspection_sources = \
	$(GEGL_introspectable_headers) \
	gegl-introspection-support.h \
	opencl/gegl-cl-introspection-support.h


Gegl-@GEGL_API_VERSION@.gir: libgegl-@GEGL_API_VERSION@.la Makefile
Gegl_@GEGL_MAJOR_VERSION@_@GEGL_MINOR_VERSION@_gir_INCLUDES = GObject-2.0 GLib-2.0
Gegl_@GEGL_MAJOR_VERSION@_@GEGL_MINOR_VERSION@_gir_CFLAGS = $(AM_CFLAGS) $(AM_CPPFLAGS)
Gegl_@GEGL_MAJOR_VERSION@_@GEGL_MINOR_VERSION@_gir_LIBS = libgegl-@GEGL_API_VERSION@.la
Gegl_@GEGL_MAJOR_VERSION@_@GEGL_MINOR_VERSION@_gir_FILES = $(introspection_sources)
INTROSPECTION_GIRS += Gegl-@GEGL_API_VERSION@.gir

girdir = $(datadir)/gir-1.0
gir_DATA = $(INTROSPECTION_GIRS)

typelibdir = $(libdir)/girepository-1.0
typelib_DATA = $(INTROSPECTION_GIRS:.gir=.typelib)

CLEANFILES += $(gir_DATA) $(typelib_DATA)

if HAVE_VALA
gegl-@GEGL_API_VERSION@.vapi: Gegl-@GEGL_API_VERSION@.gir
	$(VAPIGEN) --library=gegl-@GEGL_API_VERSION@ --metadatadir=./ Gegl-@GEGL_API_VERSION@.gir

vapidir=$(datadir)/vala/vapi
vapi_DATA = \
	gegl-@GEGL_API_VERSION@.vapi \
	gegl-@GEGL_API_VERSION@.deps

CLEANFILES += $(vapi_DATA)

endif # HAVE_VALA

EXTRA_DIST += gegl-@GEGL_API_VERSION@.deps

endif # HAVE_INTROSPECTION

#
# rules to generate built sources
#
# setup autogeneration dependencies
gen_sources = xgen-tec
CLEANFILES += $(gen_sources)

gegl-enums.c: $(srcdir)/gegl-enums.h
	glib-mkenums \
		--fhead "/* This is a generated file, do not edit directly */\n\n#include \"config.h\"\n#include <glib-object.h>\n#include \"gegl-enums.h\"" \
		--fprod "\n/* enumerations from \"@filename@\" */" \
		--vhead "GType\n@enum_name@_get_type (void)\n{\n  static GType etype = 0;\n  if (etype == 0) {\n    static const G@Type@Value values[] = {" \
		--vprod "      { @VALUENAME@, \"@VALUENAME@\", \"@valuenick@\" }," \
		--vtail "      { 0, NULL, NULL }\n    };\n    etype = g_@type@_register_static (\"@EnumName@\", values);\n  }\n  return etype;\n}\n\n" \
		$(srcdir)/gegl-enums.h > xgen-tec \
	&& cp xgen-tec $(@F) \
	&& rm -f xgen-tec
