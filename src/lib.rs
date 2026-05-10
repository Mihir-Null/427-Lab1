// CMSC427 Lab 01 — GPU Pipeline Basics  (Rust / wgpu edition)
//   Shape A — colour hexagon fan, TriangleList (6 triangles)
//   Shape B — diagonal colour band, TriangleStrip (4 triangles)
// both shapes tinted by a per-draw uniform colour (u_tint: vec4<f32>)

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub mod gpu;

use std::sync::Arc;
use wgpu::util::DeviceExt;  // create_buffer_init()
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};
use gpu::GpuCtx;

// Vertex
// 2d NDC position + RGB colour
// #[repr(C)] = C-compatible layout, required for bytemuck and VBOs since that's what we send to gpu buffer

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color:    [f32; 3],  
}

impl Vertex {
    // tells wgpu how to unpack each vert from the raw byte buffer
    // offset arithmetic: position starts at 0, color starts at byte 8 (2 x f32 x 4 bytes)
    // VAO doesn't exist directly in wgpu, instead the role is filled by the layout below 
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset:          0,
                    shader_location: 0,  // @location(0) in WGSL
                    format:          wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset:          std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,  // @location(1) in WGSL
                    format:          wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// Color Uniform
// Uniform variable in frag shader for colour
// tint is an RGBA multiplier applied to per-vertex colour in fs_main
// [1,1,1,1] = identity (no tint), anything else shifts the hue

// in WebGL this would be the uniform modified
// in wgpu we write bytes into a GPU buffer and bind it before drawing
// #[repr(C)] mandatory -> struct sent to GPU as raw bytes, tint must be at offset 0

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorUniform {
    tint: [f32; 4],  // RGBA, each in [0.0, 1.0]
}

// WGSL shader
// added:
// uniform binding @group(0) @binding(0) for u_tint
// frag shader multiplies vert colour by u_tint
// vert output carries color as vec4 so alpha is included

// @group(0) @binding(0) = "bind group slot 0, binding slot 0"
// changing the BindGroup btwn draw calls updates u_tint without rebuilding pipeline
const SHADER: &str = r#"
// uniform: one colour tint per draw call
// equivalent of a uniform var in WebGL
// bind group at @group(0) must have a buffer at @binding(0)
// matching ColorUniform layout (16 bytes, 4xf32)
@group(0) @binding(0) var<uniform> u_tint: vec4<f32>;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color:    vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       color:         vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // 2d NDC into clip space (z=0, w=1 -> no depth, no perspective)
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.color         = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // multiply per-vertex colour by the uniform tint
    // satisfies §4b: the uniform controls the final colour
    return vec4<f32>(in.color, 1.0) * u_tint;
}
"#;

// Shape data
// Shape A — Hexagon colour fan, TriangleList, 6 triangles, 18 verts
// outer ring radius ~0.42, centred at (-0.45, 0) in NDC
// centre is white, ring verts cycle through rainbow
// every 3 consecutive verts = 1 independent triangle (TriangleList)
const SHAPE_A: &[Vertex] = &[
    // triangle 0 — top
    Vertex { position: [-0.45,  0.00], color: [1.0, 1.0, 1.0] },
    //red
    Vertex { position: [-0.45,  0.42], color: [1.0, 0.2, 0.2] },
    // orange   
    Vertex { position: [-0.81,  0.21], color: [1.0, 0.5, 0.0] },
    // triangle 1 — upper-left
    Vertex { position: [-0.45,  0.00], color: [1.0, 1.0, 1.0] },
    Vertex { position: [-0.81,  0.21], color: [1.0, 0.5, 0.0] },
    // yellow
    Vertex { position: [-0.81, -0.21], color: [0.9, 0.9, 0.0] },
    // triangle 2 — lower-left
    Vertex { position: [-0.45,  0.00], color: [1.0, 1.0, 1.0] },
    Vertex { position: [-0.81, -0.21], color: [0.9, 0.9, 0.0] },
    // green
    Vertex { position: [-0.45, -0.42], color: [0.1, 0.9, 0.2] },
    // triangle 3 — bottom
    Vertex { position: [-0.45,  0.00], color: [1.0, 1.0, 1.0] },
    Vertex { position: [-0.45, -0.42], color: [0.1, 0.9, 0.2] },
    // cyan
    Vertex { position: [-0.09, -0.21], color: [0.0, 0.7, 0.9] },
    // triangle 4 — lower-right
    Vertex { position: [-0.45,  0.00], color: [1.0, 1.0, 1.0] },
    Vertex { position: [-0.09, -0.21], color: [0.0, 0.7, 0.9] },
    // violet
    Vertex { position: [-0.09,  0.21], color: [0.5, 0.0, 1.0] },
    // triangle 5 — upper-right
    Vertex { position: [-0.45,  0.00], color: [1.0, 1.0, 1.0] },
    Vertex { position: [-0.09,  0.21], color: [0.5, 0.0, 1.0] },
    Vertex { position: [-0.45,  0.42], color: [1.0, 0.2, 0.2] },
];

// Shape B — Diagonal colour band, TriangleStrip, 4 triangles, 6 verts
// TriangleStrip: N verts → N-2 triangles, each sharing 2 verts with neighbours
// winding alternates (even CCW, odd CW) so backface culling would break it => we leave cull_mode = None
// 6 verts → 4 triangles forming a diagonal band across the right half of screen
const SHAPE_B: &[Vertex] = &[
    Vertex { position: [ 0.15,  0.55], color: [0.2, 0.9, 1.0] },
    Vertex { position: [ 0.15, -0.10], color: [0.0, 0.4, 0.9] },
    Vertex { position: [ 0.55,  0.10], color: [0.7, 0.2, 1.0] },
    Vertex { position: [ 0.55, -0.55], color: [0.4, 0.0, 0.8] },
    Vertex { position: [ 0.90,  0.10], color: [1.0, 0.5, 0.8] },
    Vertex { position: [ 0.90, -0.55], color: [0.9, 0.1, 0.5] },
];

// GPU State pipelining
struct State
{
    gpu: GpuCtx,

    // two pipelines - one per primitive topology/triangle alg
    // can't switch topology mid-draw so need separate pipeline objects
    // in wgpu/WebGPU topology is baked into the pipeline object)
    render_pipeline: wgpu::RenderPipeline,  // TriangleList  - Shape A
    strip_pipeline:  wgpu::RenderPipeline,  // TriangleStrip - Shape B

    vertex_buffer_a: wgpu::Buffer,
    num_vertices_a:  u32,
    vertex_buffer_b: wgpu::Buffer,
    num_vertices_b:  u32,

    // tint bind groups - two bind groups, same layout but diff data
    tint_a_bg: wgpu::BindGroup, 
    tint_b_bg: wgpu::BindGroup, 
}

impl State
{
    pub async fn new(window: Arc<Window>) -> Self
    {
        let gpu = GpuCtx::new(window.clone()).await;

        // bind group layout for the colour uniform
        // describes what resources a bind group contains
        let color_bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Colour Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,  // @binding(0) in WGSL
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,  // fixed offset, not dynamic per draw
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        // tint buffers + bind groups
        // each holds one ColorUniform (16 bytes)
        // UNIFORM = pipeline can read as uniform, COPY_DST = can update with write_buffer
        // (not updating these each frame but using COPY_DST in case I have time to animate later)

        // Shape A tint — identity, vertex colours pass through unchanged
        let tint_a_data = ColorUniform { tint: [1.0, 1.0, 1.0, 1.0] };
        let tint_a_buf  = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Tint A Buffer"),
            contents: bytemuck::bytes_of(&tint_a_data),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let tint_a_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Tint A BG"),
            layout:  &color_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: tint_a_buf.as_entire_binding(),
            }],
        });

        // Shape B tint — blues/greens multiplied by < 1
        let tint_b_data = ColorUniform { tint: [1.0, 0.85, 0.65, 1.0] };
        let tint_b_buf  = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Tint B Buffer"),
            contents: bytemuck::bytes_of(&tint_b_data),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let tint_b_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Tint B BG"),
            layout:  &color_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: tint_b_buf.as_entire_binding(),
            }],
        });

        // shader + pipeline layout
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Lab 01 Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        // pipeline layout declares that bind group slot 0 must match color_bgl
        // both pipelines (TriangleList and TriangleStrip) share this layout, link back to comment above
    
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&color_bgl)],
            immediate_size:     0,
        });

        // render pipeline — TriangleList (Shape A)
        let render_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("TriangleList Pipeline"),
            layout: Some(&pipeline_layout),
            // VAO equivalent pt2 here, this is where translation of layout to buffer happens, check layout fn above for details
            vertex: wgpu::VertexState { 
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     gpu.format(),
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode:  None,
                ..Default::default()
            },
            depth_stencil:  None,
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // render pipeline — TriangleStrip (Shape B)
        let strip_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("TriangleStrip Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     gpu.format(),
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleStrip,  // only diff here, check if polymorphic way to impl later
                front_face: wgpu::FrontFace::Ccw,
                cull_mode:  None,
                ..Default::default()
            },
            depth_stencil:  None,
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // vertex buffers - static (we never update them), VERTEX usage only
        // equivalent to a VBO with static draw
        let vertex_buffer_a = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Shape A VBuf"),
            contents: bytemuck::cast_slice(SHAPE_A),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        let vertex_buffer_b = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Shape B VBuf"),
            contents: bytemuck::cast_slice(SHAPE_B),
            usage:    wgpu::BufferUsages::VERTEX,
        });

        Self {
            gpu,
            render_pipeline,
            strip_pipeline,
            vertex_buffer_a,
            num_vertices_a: SHAPE_A.len() as u32,
            vertex_buffer_b,
            num_vertices_b: SHAPE_B.len() as u32,
            tint_a_bg,
            tint_b_bg,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
    }

    pub fn render(&mut self) -> Result<(), ()>
    {
        // static scene so no request_redraw here
        // winit's Wait control flow handles it - only redraws on resize or OS expose events
        if !self.gpu.is_configured { return Ok(()); }

        let output = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(o)
            | wgpu::CurrentSurfaceTexture::Suboptimal(o) => o,
            wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost => {
                self.gpu.surface.configure(&self.gpu.device, &self.gpu.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
            wgpu::CurrentSurfaceTexture::Validation  => return Err(()),
        };
        let view   = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") }
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    depth_slice:    None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.08, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set:      None,
                timestamp_writes:         None,
                multiview_mask:           None,
            });

            // bind groups for tinting shapes
            rpass.set_bind_group(0, &self.tint_a_bg, &[]);
            rpass.set_pipeline(&self.render_pipeline);  // TriangleList
            rpass.set_vertex_buffer(0, self.vertex_buffer_a.slice(..));
            rpass.draw(0..self.num_vertices_a, 0..1);

            rpass.set_bind_group(0, &self.tint_b_bg, &[]);  // warm amber tint
            rpass.set_pipeline(&self.strip_pipeline);         // TriangleStrip
            rpass.set_vertex_buffer(0, self.vertex_buffer_b.slice(..));
            rpass.draw(0..self.num_vertices_b, 0..1);
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

// App (OS handling)

pub struct App
{
    #[cfg(not(target_arch = "wasm32"))]
    state: Option<State>,
    #[cfg(target_arch = "wasm32")]
    state: std::rc::Rc<std::cell::RefCell<Option<State>>>,
}

impl App
{
    pub fn new() -> Self
    {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            state: None,
            #[cfg(target_arch = "wasm32")]
            state: std::rc::Rc::new(std::cell::RefCell::new(None)),
        }
    }

    fn with_state<R>(&mut self, f: impl FnOnce(&mut State) -> R) -> Option<R>
    {
        #[cfg(not(target_arch = "wasm32"))]
        { self.state.as_mut().map(f) }
        #[cfg(target_arch = "wasm32")]
        { self.state.borrow_mut().as_mut().map(f) }
    }
}

impl ApplicationHandler for App
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop)
    {
        if self.with_state(|_| ()).is_some() { return; }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("CMSC427 Lab 01 – GPU Pipeline")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1280u32, 720u32))
                )
                .unwrap(),
        );

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("canvas-host")
                    .or_else(|| d.body().map(|b| b.into())))
                .and_then(|host| window.canvas()
                    .and_then(|c| host.append_child(&c).ok()));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(State::new(window)));
        }

        #[cfg(target_arch = "wasm32")]
        {
            let cell   = self.state.clone();
            let window = window.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let mut state = State::new(window.clone()).await;
                // on WASM, Resized never fires at startup — configure surface here while we have the window
                let s = window.inner_size();
                state.gpu.resize(s.width, s.height);
                *cell.borrow_mut() = Some(state);
                window.request_redraw();
            });
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event:      WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested  => event_loop.exit(),
            WindowEvent::Resized(size)   => {
                self.with_state(|state| state.resize(size.width, size.height));
            }
            WindowEvent::RedrawRequested => {
                self.with_state(|state| { let _ = state.render(); });
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state:        key_state,
                    ..
                },
                ..
            } => {
                if code == KeyCode::Escape && key_state.is_pressed() {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}

// Entry points

pub fn run()
{
    let event_loop = EventLoop::new().unwrap();
    let mut app    = App::new();
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start()
{
    console_log::init_with_level(log::Level::Warn).ok();
    console_error_panic_hook::set_once();

    use winit::platform::web::EventLoopExtWebSys;
    let event_loop = EventLoop::new().unwrap();
    event_loop.spawn_app(App::new());
}
