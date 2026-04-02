import os
import logging
import time

import gradio as gr

from longstream.demo import BRANCH_OPTIONS, create_demo_session, load_metadata
from longstream.demo.backend import load_frame_previews
from longstream.demo.export import export_glb
from longstream.demo.viewer import (
    build_frame_outputs,
    build_frame_outputs_from_runpy,
    build_interactive_figure,
    build_interactive_figure_from_runpy,
)
from longstream.demo.runpy_loader import (
    detect_sequences,
    load_runpy_session,
)


logger = logging.getLogger("longstream.demo")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[demo] %(message)s"))
    logger.addHandler(handler)


DEFAULT_KEYFRAME_STRIDE = 8
DEFAULT_REFRESH = 3
DEFAULT_WINDOW_SIZE = 48
DEFAULT_CHECKPOINT = os.getenv("LONGSTREAM_CHECKPOINT", "checkpoints/50_longstream.pt")


def _run_stable_demo(
    image_dir,
    uploaded_files,
    uploaded_video,
    checkpoint,
    device,
    mode,
    streaming_mode,
    refresh,
    window_size,
    compute_sky,
    branch_label,
    show_cameras,
    mask_sky,
    camera_scale,
    point_size,
    opacity,
    preview_max_points,
    glb_max_points,
):
    t0 = time.time()
    if not image_dir and not uploaded_files and not uploaded_video:
        raise gr.Error("Provide an image folder, upload images, or upload a video.")
    logger.info("Starting demo session: mode=%s, device=%s, checkpoint=%s", mode, device, checkpoint)
    session_dir = create_demo_session(
        image_dir=image_dir or "",
        uploaded_files=uploaded_files,
        uploaded_video=uploaded_video,
        checkpoint=checkpoint,
        device=device,
        mode=mode,
        streaming_mode=streaming_mode,
        keyframe_stride=DEFAULT_KEYFRAME_STRIDE,
        refresh=int(refresh),
        window_size=int(window_size),
        compute_sky=bool(compute_sky),
    )
    logger.info("Demo session created: %s", session_dir)
    logger.info("Building interactive figure...")
    fig = build_interactive_figure(
        session_dir=session_dir,
        branch=branch_label,
        display_mode="All Frames",
        frame_index=0,
        point_size=float(point_size),
        opacity=float(opacity),
        preview_max_points=int(preview_max_points),
        show_cameras=bool(show_cameras),
        camera_scale=float(camera_scale),
        mask_sky=bool(mask_sky),
    )
    logger.info("Building GLB export...")
    glb_path = export_glb(
        session_dir=session_dir,
        branch=branch_label,
        display_mode="All Frames",
        frame_index=0,
        mask_sky=bool(mask_sky),
        show_cameras=bool(show_cameras),
        camera_scale=float(camera_scale),
        max_points=int(glb_max_points),
    )
    rgb, depth, frame_label = load_frame_previews(session_dir, 0)
    meta = load_metadata(session_dir)
    slider = gr.update(
        minimum=0,
        maximum=max(meta["num_frames"] - 1, 0),
        value=0,
        step=1,
        interactive=meta["num_frames"] > 1,
    )
    sky_msg = ""
    if meta.get("has_sky_masks"):
        removed = float(meta.get("sky_removed_ratio") or 0.0) * 100.0
        sky_msg = f" | sky_removed={removed:.1f}%"
    elapsed = time.time() - t0
    status = f"Ready: {meta['num_frames']} frames | branch={branch_label}{sky_msg} ({elapsed:.1f}s)"
    logger.info("Demo session ready in %.1fs: %d frames", elapsed, meta["num_frames"])
    return (
        fig,
        glb_path,
        session_dir,
        None,
        rgb,
        depth,
        frame_label,
        slider,
        status,
    )


def _update_stable_scene(
    session_dir,
    branch_label,
    show_cameras,
    mask_sky,
    camera_scale,
    point_size,
    opacity,
    preview_max_points,
    glb_max_points,
):
    if not session_dir or not os.path.isdir(session_dir):
        return None, None, "Run reconstruction first."
    logger.info("Updating demo scene: branch=%s", branch_label)
    t0 = time.time()
    fig = build_interactive_figure(
        session_dir=session_dir,
        branch=branch_label,
        display_mode="All Frames",
        frame_index=0,
        point_size=float(point_size),
        opacity=float(opacity),
        preview_max_points=int(preview_max_points),
        show_cameras=bool(show_cameras),
        camera_scale=float(camera_scale),
        mask_sky=bool(mask_sky),
    )
    logger.info("Exporting GLB...")
    glb_path = export_glb(
        session_dir=session_dir,
        branch=branch_label,
        display_mode="All Frames",
        frame_index=0,
        mask_sky=bool(mask_sky),
        show_cameras=bool(show_cameras),
        camera_scale=float(camera_scale),
        max_points=int(glb_max_points),
    )
    meta = load_metadata(session_dir)
    sky_msg = ""
    if meta.get("has_sky_masks"):
        removed = float(meta.get("sky_removed_ratio") or 0.0) * 100.0
        sky_msg = f" | sky_removed={removed:.1f}%"
    elapsed = time.time() - t0
    logger.info("Scene updated in %.1fs", elapsed)
    return fig, glb_path, f"Updated preview: {branch_label}{sky_msg} ({elapsed:.1f}s)"


def _update_frame_preview(session_dir, frame_index):
    if not session_dir or not os.path.isdir(session_dir):
        return None, None, ""
    rgb, depth, label = load_frame_previews(session_dir, int(frame_index))
    return rgb, depth, label


def _update_sequences_list(output_root):
    if not output_root or not os.path.isdir(output_root):
        logger.info("No valid output root provided")
        return gr.update(choices=[], value=None)
    logger.info("Scanning sequences in: %s", output_root)
    seqs = detect_sequences(output_root)
    if not seqs:
        logger.info("No sequences found in %s", output_root)
        return gr.update(choices=[], value=None)
    logger.info("Found %d sequences: %s", len(seqs), ", ".join(seqs))
    return gr.update(choices=seqs, value=seqs[0])


def _load_runpy_output(
    output_root,
    seq_name,
    branch_label,
    show_cameras,
    mask_sky,
    camera_scale,
    point_size,
    opacity,
    preview_max_points,
    glb_max_points,
):
    t0 = time.time()
    if not output_root or not os.path.isdir(output_root):
        raise gr.Error("Provide a valid output root directory.")
    if not seq_name:
        raise gr.Error("Select a sequence from the dropdown.")

    seq_dir = os.path.join(output_root, seq_name)
    logger.info("Loading run.py output: seq_dir=%s", seq_dir)
    session = load_runpy_session(seq_dir)
    meta = session["metadata"]
    logger.info(
        "Loaded sequence: %d frames, %dx%d, sky_masks=%s, point_sources=%s",
        meta["num_frames"], meta["width"], meta["height"],
        meta["has_sky_masks"], meta["point_sources"],
    )

    logger.info("Building interactive figure...")
    fig = build_interactive_figure_from_runpy(
        session=session,
        branch=branch_label,
        display_mode="All Frames",
        frame_index=0,
        point_size=float(point_size),
        opacity=float(opacity),
        preview_max_points=int(preview_max_points),
        show_cameras=bool(show_cameras),
        camera_scale=float(camera_scale),
        mask_sky=bool(mask_sky),
    )

    logger.info("Exporting GLB...")
    glb_path = _export_glb_from_runpy(
        session=session,
        branch=branch_label,
        display_mode="All Frames",
        frame_index=0,
        mask_sky=bool(mask_sky),
        show_cameras=bool(show_cameras),
        camera_scale=float(camera_scale),
        max_points=int(glb_max_points),
    )

    logger.info("Loading frame previews...")
    rgb, depth, frame_label = build_frame_outputs_from_runpy(session, 0)
    slider = gr.update(
        minimum=0,
        maximum=max(meta["num_frames"] - 1, 0),
        value=0,
        step=1,
        interactive=meta["num_frames"] > 1,
    )
    sky_msg = ""
    if meta.get("has_sky_masks"):
        removed = float(meta.get("sky_removed_ratio") or 0.0) * 100.0
        sky_msg = f" | sky_removed={removed:.1f}%"
    elapsed = time.time() - t0
    status = f"Loaded: {seq_name} | {meta['num_frames']} frames | branch={branch_label}{sky_msg} ({elapsed:.1f}s)"
    logger.info("Run.py output loaded in %.1fs", elapsed)
    return (
        fig,
        glb_path,
        None,
        session,
        rgb,
        depth,
        frame_label,
        slider,
        status,
    )


def _update_runpy_scene(
    session,
    branch_label,
    show_cameras,
    mask_sky,
    camera_scale,
    point_size,
    opacity,
    preview_max_points,
    glb_max_points,
):
    if session is None:
        return None, None, "Load a run.py output first."
    logger.info("Updating runpy scene: branch=%s", branch_label)
    t0 = time.time()
    fig = build_interactive_figure_from_runpy(
        session=session,
        branch=branch_label,
        display_mode="All Frames",
        frame_index=0,
        point_size=float(point_size),
        opacity=float(opacity),
        preview_max_points=int(preview_max_points),
        show_cameras=bool(show_cameras),
        camera_scale=float(camera_scale),
        mask_sky=bool(mask_sky),
    )
    logger.info("Exporting GLB...")
    glb_path = _export_glb_from_runpy(
        session=session,
        branch=branch_label,
        display_mode="All Frames",
        frame_index=0,
        mask_sky=bool(mask_sky),
        show_cameras=bool(show_cameras),
        camera_scale=float(camera_scale),
        max_points=int(glb_max_points),
    )
    meta = session["metadata"]
    sky_msg = ""
    if meta.get("has_sky_masks"):
        removed = float(meta.get("sky_removed_ratio") or 0.0) * 100.0
        sky_msg = f" | sky_removed={removed:.1f}%"
    elapsed = time.time() - t0
    logger.info("Runpy scene updated in %.1fs", elapsed)
    return fig, glb_path, f"Updated preview: {branch_label}{sky_msg} ({elapsed:.1f}s)"


def _update_runpy_frame_preview(session, frame_index):
    if session is None:
        return None, None, ""
    rgb, depth, label = build_frame_outputs_from_runpy(session, int(frame_index))
    return rgb, depth, label


def _export_glb_from_runpy(
    session,
    branch,
    display_mode,
    frame_index,
    mask_sky,
    show_cameras,
    camera_scale,
    max_points,
):
    import tempfile

    import numpy as np
    import trimesh

    from longstream.demo.runpy_loader import collect_points_from_runpy

    logger.info("Collecting points: branch=%s, max_points=%d, mask_sky=%s", branch, max_points, mask_sky)
    t0 = time.time()
    points, colors, _ = collect_points_from_runpy(
        session=session,
        branch=branch,
        display_mode=display_mode,
        frame_index=frame_index,
        mask_sky=mask_sky,
        max_points=max_points,
        seed=frame_index,
    )

    if len(points) == 0:
        logger.warning("No points collected for branch=%s", branch)
        return None

    logger.info("Collected %d points in %.1fs, building scene...", len(points), time.time() - t0)
    cloud = trimesh.PointCloud(vertices=points, colors=colors)
    scene = trimesh.Scene([cloud])

    if show_cameras:
        from longstream.demo.runpy_loader import camera_geometry_from_runpy

        logger.info("Adding camera geometry...")
        centers, frustums, _ = camera_geometry_from_runpy(
            session=session,
            display_mode=display_mode,
            frame_index=frame_index,
            camera_scale_ratio=camera_scale,
            points_hint=points,
        )
        logger.info("Adding %d camera frustums", len(frustums))
        for center, corners in frustums:
            line_segments = [
                (corners[i], corners[j])
                for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]
            ]
            line_segments.extend([(center, corners[i]) for i in range(4)])
            for a, b in line_segments:
                line = trimesh.load_path(np.stack([a, b], axis=0))
                scene.add_geometry(line)

    path = os.path.join(tempfile.gettempdir(), f"longstream_{os.getpid()}.glb")
    logger.info("Exporting GLB to %s...", path)
    t1 = time.time()
    scene.export(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info("GLB exported in %.1fs (%.1f MB)", time.time() - t1, size_mb)
    return path


def main():
    with gr.Blocks(title="LongStream Demo") as demo:
        session_dir = gr.Textbox(visible=False)
        runpy_session = gr.State(None)
        active_mode = gr.State("demo")

        gr.Markdown("# LongStream Demo")

        with gr.Row():
            viz_mode = gr.Radio(
                label="Visualization Mode",
                choices=["Demo Mode", "Load run.py Output"],
                value="Demo Mode",
            )

        with gr.Group(visible=True) as demo_input_group:
            with gr.Row():
                image_dir = gr.Textbox(
                    label="Image Folder", placeholder="/path/to/sequence"
                )
                uploaded_files = gr.File(
                    label="Upload Images", file_count="multiple", file_types=["image"]
                )
                uploaded_video = gr.File(
                    label="Upload Video", file_count="single", file_types=["video"]
                )

            with gr.Row():
                checkpoint = gr.Textbox(label="Checkpoint", value=DEFAULT_CHECKPOINT)
                device = gr.Dropdown(
                    label="Device", choices=["cuda", "cpu"], value="cuda"
                )

            with gr.Accordion("Inference", open=False):
                with gr.Row():
                    mode = gr.Dropdown(
                        label="Mode",
                        choices=["streaming_refresh", "batch_refresh"],
                        value="batch_refresh",
                    )
                    streaming_mode = gr.Dropdown(
                        label="Streaming Mode",
                        choices=["causal", "window"],
                        value="causal",
                    )
                with gr.Row():
                    refresh = gr.Slider(
                        label="Refresh",
                        minimum=2,
                        maximum=9,
                        step=1,
                        value=DEFAULT_REFRESH,
                    )
                    window_size = gr.Slider(
                        label="Window Size",
                        minimum=1,
                        maximum=64,
                        step=1,
                        value=DEFAULT_WINDOW_SIZE,
                    )
                    compute_sky = gr.Checkbox(label="Compute Sky Masks", value=True)

        with gr.Group(visible=False) as runpy_input_group:
            output_root = gr.Textbox(
                label="Output Root Directory",
                placeholder="/path/to/outputs",
            )
            with gr.Row():
                seq_selector = gr.Dropdown(
                    label="Sequence",
                    choices=[],
                    value=None,
                )
                refresh_seqs_btn = gr.Button("Refresh Sequences", size="sm")

        with gr.Accordion("GLB Settings", open=True):
            with gr.Row():
                branch_label = gr.Dropdown(
                    label="Point Cloud Branch",
                    choices=BRANCH_OPTIONS,
                    value="Point Head + Pose",
                )
                show_cameras = gr.Checkbox(label="Show Cameras", value=True)
                mask_sky = gr.Checkbox(label="Mask Sky", value=True)
            with gr.Row():
                point_size = gr.Slider(
                    label="Point Size",
                    minimum=0.05,
                    maximum=2.0,
                    step=0.05,
                    value=0.3,
                )
                opacity = gr.Slider(
                    label="Opacity",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=0.75,
                )
                preview_max_points = gr.Slider(
                    label="Preview Max Points",
                    minimum=5000,
                    maximum=1000000,
                    step=10000,
                    value=100000,
                )
            with gr.Row():
                camera_scale = gr.Slider(
                    label="Camera Scale",
                    minimum=0.001,
                    maximum=0.05,
                    step=0.001,
                    value=0.01,
                )
                glb_max_points = gr.Slider(
                    label="GLB Max Points",
                    minimum=20000,
                    maximum=1000000,
                    step=10000,
                    value=400000,
                )

        with gr.Row():
            run_btn = gr.Button("Run Stable Demo", variant="primary")
            load_btn = gr.Button("Load Output", variant="primary", visible=False)

        status = gr.Markdown("Provide input images, then run reconstruction.")

        plot = gr.Plot(label="Scene Preview")

        glb_file = gr.File(label="Download GLB")

        with gr.Row():
            frame_slider = gr.Slider(
                label="Preview Frame",
                minimum=0,
                maximum=0,
                step=1,
                value=0,
                interactive=False,
            )
            frame_label = gr.Textbox(label="Frame")
        with gr.Row():
            rgb_preview = gr.Image(label="RGB", type="numpy")
            depth_preview = gr.Image(label="Depth Plasma", type="numpy")

        def _toggle_viz_mode(choice):
            logger.info("Switching visualization mode: %s", choice)
            if choice == "Demo Mode":
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "demo",
                )
            else:
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    "runpy",
                )

        viz_mode.change(
            _toggle_viz_mode,
            inputs=[viz_mode],
            outputs=[
                demo_input_group,
                runpy_input_group,
                run_btn,
                load_btn,
                active_mode,
            ],
        )

        refresh_seqs_btn.click(
            _update_sequences_list,
            inputs=[output_root],
            outputs=[seq_selector],
        )

        run_btn.click(
            _run_stable_demo,
            inputs=[
                image_dir,
                uploaded_files,
                uploaded_video,
                checkpoint,
                device,
                mode,
                streaming_mode,
                refresh,
                window_size,
                compute_sky,
                branch_label,
                show_cameras,
                mask_sky,
                camera_scale,
                point_size,
                opacity,
                preview_max_points,
                glb_max_points,
            ],
            outputs=[
                plot,
                glb_file,
                session_dir,
                runpy_session,
                rgb_preview,
                depth_preview,
                frame_label,
                frame_slider,
                status,
            ],
        )

        load_btn.click(
            _load_runpy_output,
            inputs=[
                output_root,
                seq_selector,
                branch_label,
                show_cameras,
                mask_sky,
                camera_scale,
                point_size,
                opacity,
                preview_max_points,
                glb_max_points,
            ],
            outputs=[
                plot,
                glb_file,
                session_dir,
                runpy_session,
                rgb_preview,
                depth_preview,
                frame_label,
                frame_slider,
                status,
            ],
        )

        for component in [
            branch_label,
            show_cameras,
            mask_sky,
            camera_scale,
            point_size,
            opacity,
            preview_max_points,
            glb_max_points,
        ]:
            component.change(
                _update_stable_scene,
                inputs=[
                    session_dir,
                    branch_label,
                    show_cameras,
                    mask_sky,
                    camera_scale,
                    point_size,
                    opacity,
                    preview_max_points,
                    glb_max_points,
                ],
                outputs=[plot, glb_file, status],
            ).then(
                _update_runpy_scene,
                inputs=[
                    runpy_session,
                    branch_label,
                    show_cameras,
                    mask_sky,
                    camera_scale,
                    point_size,
                    opacity,
                    preview_max_points,
                    glb_max_points,
                ],
                outputs=[plot, glb_file, status],
            )

        frame_slider.change(
            _update_frame_preview,
            inputs=[session_dir, frame_slider],
            outputs=[rgb_preview, depth_preview, frame_label],
        ).then(
            _update_runpy_frame_preview,
            inputs=[runpy_session, frame_slider],
            outputs=[rgb_preview, depth_preview, frame_label],
        )

    demo.launch(
        allowed_paths=[os.path.abspath("outputs")],
    )


if __name__ == "__main__":
    main()
