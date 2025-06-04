# Rendering the Input/Results with Blender

All the 3D contents within our paper is rendered with Blender's Python module `bpy`.

## Dependencies

To render similar images to the paper, you should use `bpy>=4.0.0` (4.0.0 and 4.3.0 are tested on our own machine). By default, `bpy` is included as a dependency when building the docker image, so you can render images in the container.

However, Blender's EEVEE render engine (hardware-accelerated rasterization render engine) is inacceptably slow in a docker container because EEVEE requires a monitor (screen) to enable OpenGL/Vulkan acceleration. Definitely you can use Cycles to preform path tracing, but if you want to render some results faster just for examining, you may want to use EEVEE outside the container. To achieve this, just install
```
bpy>=4.0.0 numpy
```

## Scripts

All Python scripts in this directory are about configuring and using Blender. Each file named with a prefix `render_` is an entry point, including
- *render_input.py*: Rendering input pointclouds
- *render_gt.py*: Rendering the GT meshes
- *render_eval.py*: Rendering results saved by *test_shapenet.py*
- *render_progress.py*: Rendering a mesh sequence saved by *reconstructed.py* (with `--save-progress` argument passed)
- *render_nc.py*: Visualizing Normal Consistency of the reconstructed surface saved by *test_shapenet.py*
- *render_tangent_samples.py*: Render query points sampled for tangent error metric

By default, most of these scripts uses the Cycles render engine, which performs path tracing. All visible CUDA devices will be used (see function `init_scene` in *prepare.py*) for acceleration, you can use the environment variable `CUDA_VISIBLE_DEVICES` to limit the GPU used for rendering.

> Note: *render_input.py*, *render_nc.py* and *render_tangent_samples.py* only support EEVEE, because using path-tracing to render pointcloud may result in too much shadow. It is not recommended to run *render_nc.py* in a container because it heavily depends on hardware graphics API (e.g. OpenGL or Vulkan) to render a large amount of points.
