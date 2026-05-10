# CMSC427 Lab 01 Report

This lab was implemented in Rust with wgpu and winit. It ports the basic WebGL pipeline idea into the same  GPU model used by the coursework projects: define vertex data on the CPU, upload it into GPU buffers, create a render pipeline, and issue draw calls. Much of the infrastructure has been ported over from lab 0 (because I wrote that after this lab)

The general pipeline for in webgpu is similar to webgl, with every piece of geometry being processed as follows:
CPU (Rust) → VBO (GPU buffer) → vertex shader → rasteriser → fragment shader → framebuffer

VBO: blob of raw bytes on the GPU. we tell the pipeline its layout when defining the vertex in wgpu (strides, offsets, attr formats) so the vertex shader can unpack each vert

vertex shader: runs once per vert, transforms position + passes colour through

rasteriser: given the primitive topology, determines which pixels each triangle covers and interpolates vertex attributes across them, standard


frag shader: runs once per covered pixel, outputs final colour this is where the uniform tint/variable is applied.

The core difference moving from webgl to wgpu is that the uniform vars (constant across all shader invocations for a draw call) live in GPU buffers and are accessed through a bind group. Xhanging the bound bind group betweenn draw calls = changing what the shader sees.

Again, I used the same [learn-wgpu tutorial](https://sotrh.github.io/learn-wgpu/) reference throughout and AI tools for understanding concepts only. VSCode extensions were used for linting.


## Vertex Data

The project defines a small vertex type with a two-dimensional position and an RGB color:

```rust
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}
```

The vertex buffer layout maps position to shader location `0` and color to shader location `1`. This is the wgpu equivalent of setting up WebGL vertex attributes. The important detail is that the Rust struct layout, the `VertexBufferLayout`, and the WGSL `VertexInput` struct all agree about the order and size of the attributes.

## Shader Pipeline

The WGSL vertex shader copies each two-dimensional vertex position into clip space by creating a 4d vector with `z = 0.0` and `w = 1.0` (therefore defining only x and y). The fragment shader returns the interpolated vertex color which is then interpolated across the triangles.

## Triangle List Shape

The first object, `SHAPE_A`, is rendered via a `TriangleList` pipeline instead of the less modern triangle fan. It is a six-triangle color fan around a center point on the left side of the window. Each triangle repeats the center vertex and uses two adjacent outer vertices.

This demonstrates the most direct primitive mode: every group of three vertices forms one independent triangle. The color fan makes interpolation easy to see because colors blend from the white center to the colored outer ring.

## Triangle Strip Shape

The second object, `SHAPE_B`, is rendered with a separate `TriangleStrip` pipeline that matches what was learned in class. It uses six vertices to create a diagonal band on the right side of the window.

This demonstrates how the strip reuses previous vertices. After the first triangle, each additional vertex contributes one more triangle. The result is a compact way to describe connected geometry, and it makes the difference between `TriangleList` and `TriangleStrip` visible.

## wgpu Porting Notes

In WebGL, much of the pipeline is controlled by mutable global state. In this wgpu version, the pipeline state is explicit. The project creates two render pipelines because the primitive topology is part of pipeline state: one pipeline uses `TriangleList`, and the other uses `TriangleStrip`. I chose to use TriangleList in place of the Triangle Fan algorithm we learned in class as it was the most optimized/recommended when looking at comparisons of wgpu algorithms online (with triangle fans being very slow/unoptimized according to most gyudes)

## Result

The final scene shows two different pieces of geometry in one render pass. The left shape demonstrates independent triangle submission, while the right shape demonstrates connected strip submission. Together they show a full gpu pipeline walkthrough: vertex buffers, shader inputs, primitive topology, render pipelines, and draw calls are all visible in a small wgpu program.
