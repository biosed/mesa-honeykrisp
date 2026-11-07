#version 460
#extension GL_EXT_mesh_shader : require

layout(location = 0) in vec4 vertex_colour;
perprimitiveEXT layout(location = 1) flat in vec4 primitive_colour;

void
main()
{
   /* Keep both interface classes live in an attachmentless no-submit test. */
   if (vertex_colour.x + primitive_colour.y < -1.0)
      discard;
}
