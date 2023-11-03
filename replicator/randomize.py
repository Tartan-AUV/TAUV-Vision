import omni.replicator.core as rep

with rep.new_layer():
    camera = rep.create.camera(position=(0, 0, 1000))
    render_product = rep.create.render_product(camera, (1280, 720))


    def move_distractors():
        distractors = rep.get.prims(semantics=[('type', 'distractor')])

        with distractors:
            rep.modify.pose(
                position=rep.distribution.uniform((-500, -500, -500), (500, 500, 0)),
                rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)),
            )

        return distractors.node


    rep.randomizer.register(move_distractors)


    def move_objects():
        objects = rep.get.prims(semantics=[('type', 'object')])

        with objects:
            rep.modify.pose(
                position=rep.distribution.uniform((-100, -100, 0), (100, 100, 700)),
                rotation=rep.distribution.uniform((-30, -30, -180), (30, 30, 180)),
            )

        return objects.node


    rep.randomizer.register(move_objects)

    with rep.trigger.on_frame(num_frames=10000):
        rep.randomizer.move_distractors()
        rep.randomizer.move_objects()

basic_writer = rep.WriterRegistry.get("BasicWriter")
basic_writer.initialize(
    output_dir=f"~/Documents/replicator_out/",
    rgb=True,
    bounding_box_2d_loose=True,
    bounding_box_2d_tight=True,
    bounding_box_3d=True,
    distance_to_camera=True,
    distance_to_image_plane=True,
    instance_segmentation=True,
    semantic_segmentation=True,
    camera_params=True,
)

basic_writer.attach(render_product)