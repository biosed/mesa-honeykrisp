#version 460
#extension GL_EXT_mesh_shader : require

layout(location = 0) in vec4 vertex_colour;
layout(location = 0) out vec4 colour;

void
main()
{
   colour = vertex_colour;
}
