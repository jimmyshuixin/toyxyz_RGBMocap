"""Material semantics that differ between MMD and ordinary Blender surfaces."""
from types import SimpleNamespace as NS
import unittest

from mocap.retarget.blender_materials import base_surface


class Nodes(dict):
    def __iter__(self):
        return iter(self.values())


def socket(value, node=None):
    return NS(default_value=value, is_linked=node is not None,
              links=[] if node is None else [NS(from_node=node)])


def image_node(image):
    return NS(type='TEX_IMAGE', image=image, name=image)


def material(nodes):
    return NS(diffuse_color=[.8, .8, .8, .4], use_nodes=True, node_tree=NS(nodes=Nodes(nodes)))


class CharacterMaterialTests(unittest.TestCase):
    def test_mmd_hidden_textured_surface_keeps_zero_alpha(self):
        mat = material({'mmd_shader': NS(inputs={'Diffuse Color': socket([.2, .4, .6, 1]),
                                               'Alpha': socket(0.)}),
                        'mmd_base_tex': image_node('base.png')})
        self.assertEqual(base_surface(mat), ([.2, .4, .6, 0.], 'base.png'))

    def test_mmd_sphere_and_toon_never_become_base_image(self):
        mat = material({'mmd_shader': NS(inputs={}),
                        'mmd_sphere_tex': image_node('sphere.png'),
                        'mmd_toon_tex': image_node('toon.bmp')})
        self.assertEqual(base_surface(mat), ([.8, .8, .8, .4], None))

    def test_mmd_base_selection_does_not_depend_on_node_order(self):
        mat = material({'mmd_sphere_tex': image_node('sphere.png'),
                        'mmd_shader': NS(inputs={}),
                        'mmd_base_tex': image_node('base.png')})
        self.assertEqual(base_surface(mat)[1], 'base.png')

    def test_principled_link_replaces_default_color_but_keeps_alpha(self):
        image = image_node('base.png')
        mat = material({'bsdf': NS(type='BSDF_PRINCIPLED', inputs={
            'Base Color': socket([.8, .8, .8, 1.], image), 'Alpha': socket(.25)})})
        self.assertEqual(base_surface(mat), ([1., 1., 1., .25], 'base.png'))

    def test_principled_unlinked_base_does_not_select_unconnected_image(self):
        mat = material({'bsdf': NS(type='BSDF_PRINCIPLED', inputs={
            'Base Color': socket([.2, .3, .4, 1.]), 'Alpha': socket(.5)}),
            'unused': image_node('unused.png')})
        self.assertEqual(base_surface(mat), ([.2, .3, .4, .5], None))


if __name__ == '__main__':
    unittest.main()
