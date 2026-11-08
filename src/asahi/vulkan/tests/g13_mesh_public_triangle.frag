#version 460
#extension GL_EXT_mesh_shader : require

layout(location = 0) in vec4 vertex_colour;
perprimitiveEXT layout(location = 1) flat in vec4 primitive_colour;
layout(location = 0) out vec4 colour;

void
main()
{
   /* Keep both interface classes live.  The public fixture has no attachment,
    * while the physical readback fixture consumes this exact colour. */
   colour = vec4(vertex_colour.x, primitive_colour.y, 0.75, 1.0);
}
