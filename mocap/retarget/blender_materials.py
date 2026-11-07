"""Resolve the supported base-color surface without mistaking MMD effect maps."""


def base_surface(material):
    """Return RGBA multiplier and base image for the simplified character view.

    MMD sphere/toon images are lighting inputs, never diffuse UV textures.
    A directly linked Principled image replaces that socket's default color;
    MMD's diffuse color and material alpha instead multiply its base image.
    """
    if material is None:
        return [.65, .65, .65, 1.], None
    color = list(material.diffuse_color)
    if not material.use_nodes:
        return color, None
    nodes = material.node_tree.nodes
    mmd = nodes.get('mmd_shader')
    if mmd is not None:
        diffuse, alpha = mmd.inputs.get('Diffuse Color'), mmd.inputs.get('Alpha')
        if diffuse is not None:
            color[:3] = diffuse.default_value[:3]
        if alpha is not None:
            color[3] = float(alpha.default_value)
        node = nodes.get('mmd_base_tex')
        return color, node.image if node is not None and node.type == 'TEX_IMAGE' else None
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is not None:
        base, alpha = bsdf.inputs.get('Base Color'), bsdf.inputs.get('Alpha')
        if base is not None:
            color = list(base.default_value)
            if alpha is not None:
                color[3] = 1. if alpha.is_linked else float(alpha.default_value)
            if base.is_linked:
                node = base.links[0].from_node
                if node.type == 'TEX_IMAGE' and node.image:
                    color[:3] = [1., 1., 1.]
                    return color, node.image
            else:
                return color, None
    # Legacy VRM/group shaders: retain the existing base-image approximation,
    # while excluding named shading inputs and preserving material opacity.
    node = next((n for n in nodes if n.type == 'TEX_IMAGE' and n.image
                 and not any(tag in n.name.lower() for tag in ('shade', 'sphere', 'toon', 'normal'))), None)
    if node is not None:
        color[:3] = [1., 1., 1.]
    return color, node.image if node is not None else None
