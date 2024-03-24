import omni.replicator.core as rep
import sys
import asyncio
import carb.settings

# ~/.local/share/ov/pkg/code-2022.3.3/omni.code.sh --no-window --/omni/replicator/script=/home/theo/Documents/yolo_pose/replicator/randomize.py

# Camera params fails if this is turned on
NUM_FRAMES = 10000

SCENE_PRIM_PREFIX = "/Replicator/Ref_Xform/Ref"

# carb.settings.get_settings().set("/omni/replicator/RTSubframes", 8)

with rep.new_layer():
    scene = rep.create.from_usd("/home/theo/Documents/yolo_pose/models/underwater_scene_1/underwater_scene_1.usd")

    camera = rep.create.camera(
        position=(0, 0, 0),
        rotation=(0, 0, 0),
    )
    render_product = rep.create.render_product(camera, (640, 360))

    def randomize_sky():
        print("randomizing sky")
        sky = rep.get.prims(f"{SCENE_PRIM_PREFIX}/Environment/sky")

        with sky:
            rep.modify.pose(
                rotation=rep.distribution.uniform((270, -180, 0), (270, 180, 0)),
            )
            rep.modify.attribute("texture:file", rep.distribution.choice([
                "/home/theo/Documents/dosch_design_underwater_hdri/spherical_map/Underwater-09_XXL.hdr",
                "/home/theo/Documents/dosch_design_underwater_hdri/spherical_map/Underwater-10_XXL.hdr",
                "/home/theo/Documents/dosch_design_underwater_hdri/spherical_map/Underwater-23_XXL.hdr",
                "/home/theo/Documents/dosch_design_underwater_hdri/spherical_map/Underwater-24_XXL.hdr",
                "/home/theo/Documents/dosch_design_underwater_hdri/spherical_map/Underwater-17_XXL.hdr",
                "/home/theo/Documents/dosch_design_underwater_hdri/spherical_map/Underwater-18_XXL.hdr",
                "/home/theo/Documents/dosch_design_underwater_hdri/spherical_map/Underwater-19_XXL.hdr",
                "/home/theo/Documents/dosch_design_underwater_hdri/spherical_map/Underwater-20_XXL.hdr",
                "/home/theo/Documents/auto_service_4k.exr",
                "/home/theo/Documents/kart_club_4k.exr",
                "/home/theo/Documents/machine_shop_01_4k.exr",
                "/home/theo/Documents/school_quad_4k.exr",
                "/home/theo/Documents/whale_skeleton_4k.exr",
            ]))

            rep.modify.attribute("intensity", rep.distribution.uniform(200, 250))
            rep.modify.attribute("exposure", rep.distribution.uniform(0, 5))

        return sky.node

    rep.randomizer.register(randomize_sky)

    def randomize_sun():
        print("randomizing sun")
        sun = rep.get.prim_at_path(f"{SCENE_PRIM_PREFIX}/Environment/sun")

        with sun:
            rep.modify.pose(
                rotation=rep.distribution.uniform((0, -180, 0), (45, 180, 0)),
            )

            rep.modify.attribute("colorTemperature", rep.distribution.normal(6500, 1000))

            rep.modify.attribute("intensity", rep.distribution.uniform(0, 1000))

        return sun.node

    rep.randomizer.register(randomize_sun)

    def randomize_water():
        print("randomizing water")
        water = rep.get.prim_at_path(f"{SCENE_PRIM_PREFIX}/Looks/Water")

        with water:
            rep.modify.attribute("inputs:volume_scattering", rep.distribution.uniform(0.05, 0.1))
            rep.modify.attribute("inputs:base_thickness", rep.distribution.uniform(1, 5))

        return water.node

    rep.randomizer.register(randomize_water)

    def randomize_environment():
        print("randomizing environment")

        environment = rep.get.prim_at_path(f"{SCENE_PRIM_PREFIX}/Environment")

        with environment:
            rep.modify.pose(
                position=rep.distribution.uniform((0, 200, 0), (0, 1000, 0)),
            )

        return environment.node

    rep.randomizer.register(randomize_environment)

    def randomize_distractors():
        print("getting distractors")
        distractors = rep.get.prims(semantics=[('type', 'distractor')])
        print("got distractors")

        with distractors:
            rep.modify.pose_camera_relative(
                camera=camera,
                render_product=render_product,
                horizontal_location=rep.distribution.uniform(-1, 1),
                vertical_location=rep.distribution.uniform(-1, 1),
                distance=rep.distribution.uniform(600, 1000),
            )

            rep.modify.pose(
                rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)),
            )

            rep.randomizer.color(
                colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
            )

        return distractors.node

    rep.randomizer.register(randomize_distractors)

    def randomize_objects():
        print("creating objects")
        objects = rep.randomizer.instantiate([
            # "/home/theo/Documents/yolo_pose/models/torpedo_22/usd/torpedo_22_bootlegger_final_circle.usd",
            "/home/theo/Documents/yolo_pose/models/torpedo_22/usd/torpedo_22_bootlegger_final_trapezoid.usd",
            # "/home/theo/Documents/yolo_pose/models/buoy_23/usd/buoy_23_1_final.usd",
        ], size=1, mode="reference", use_cache=True)
        print("created objects")

        with objects:
            rep.modify.pose_camera_relative(
                camera=camera,
                render_product=render_product,
                horizontal_location=rep.distribution.uniform(-0.75, 0.75),
                vertical_location=rep.distribution.uniform(-0.75, 0.75),
                distance=rep.distribution.uniform(100, 1000),
            )

            rep.modify.pose(
                rotation=rep.distribution.uniform((-15, -15, -15), (15, 15, 15)),
            )

        print("done randomizing objects")

        return objects.node

    rep.randomizer.register(randomize_objects)

    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([render_product])

    bbox_annot = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    bbox_annot.attach([render_product])

    bbox3d_annot = rep.AnnotatorRegistry.get_annotator("bounding_box_3d")
    bbox3d_annot.attach([render_product])

    instance_seg_annot = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast")
    instance_seg_annot.attach([render_product])

    camera_params_annot = rep.AnnotatorRegistry.get_annotator("camera_params")
    camera_params_annot.attach([render_product])

    basic_writer = rep.BasicWriter(
        output_dir=f"/home/theo/Documents/replicator_out/",
        colorize_instance_segmentation=False,
    )

    with rep.trigger.on_frame():
        print("trigger")
        rep.randomizer.randomize_sky()
        rep.randomizer.randomize_sun()
        rep.randomizer.randomize_water()
        rep.randomizer.randomize_environment()
        rep.randomizer.randomize_distractors()
        rep.randomizer.randomize_objects()

    async def run():
        await rep.orchestrator.step_async()

        camera_params_data = camera_params_annot.get_data()
        print(f"camera_params_data: {camera_params_data}")

        print("writing camera params!")

        basic_writer.write({
            "trigger_outputs": {"on_time": 0},
            "camera_params": camera_params_data,
        })

        print("wrote camera params")

        rep.settings.set_render_pathtraced(16)

        for i in range(NUM_FRAMES):
            print(f"waiting...")
            sys.stdout.flush()
            await rep.orchestrator.step_async()
            print(f"done waiting")

            rgb_data = rgb_annot.get_data()
            print(f"rgb_data: {rgb_data}")

            bbox_data = bbox_annot.get_data()
            print(f"bbox_data: {bbox_data}")

            bbox3d_data = bbox3d_annot.get_data()
            print(f"fbbox3d_data: {bbox3d_data}")

            instance_seg_data = instance_seg_annot.get_data()
            print(f"instance_seg_data: {instance_seg_data}")

            # camera_params_data = camera_params_annot.get_data()
            # print(f"camera_params_data: {camera_params_data}")

            basic_writer.write({
                "trigger_outputs": {"on_time": 0},
                "rgb": rgb_data,
                "bounding_box_2d_tight": bbox_data,
                "bounding_box_3d": bbox3d_data,
                "instance_segmentation": instance_seg_data,
                # "camera_params": camera_params_data,
            })

    asyncio.ensure_future(run())
