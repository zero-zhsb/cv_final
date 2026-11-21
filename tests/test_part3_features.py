import unittest
import numpy as np
from src.part3.features import fuse_features


class FusionTests(unittest.TestCase):
    def test_joint_distance_is_mean_of_layer_distances(self):
        rng=np.random.default_rng(43)
        q=[rng.normal(size=(5,8)).astype('f') for _ in range(2)]
        r=[rng.normal(size=(7,8)).astype('f') for _ in range(2)]
        expected=sum(1-(a/np.linalg.norm(a,axis=1,keepdims=True))@(b/np.linalg.norm(b,axis=1,keepdims=True)).T for a,b in zip(q,r))/2
        actual=1-fuse_features(*q)@fuse_features(*r).T
        np.testing.assert_allclose(actual,expected,atol=3e-7)
        np.testing.assert_allclose(np.linalg.norm(fuse_features(*q),axis=1),1,atol=2e-7)

    def test_inputs_not_mutated_and_invalid_grids_rejected(self):
        x=np.array([[3.,4.],[5.,12.]],dtype='f');original=x.copy()
        fuse_features(x,x);np.testing.assert_array_equal(x,original)
        for layers in [(),(x,x[:1]),(x,np.zeros_like(x)),(x,np.full_like(x,np.nan))]:
            with self.assertRaises(ValueError):fuse_features(*layers)

    def test_fusion_does_not_allow_independent_reference_matches(self):
        q=np.array([[1.,0.]],dtype='f')
        a=np.array([[1.,0.],[-1.,0.]],dtype='f');b=-a
        joint=(1-fuse_features(q,q)@fuse_features(a,b).T).min()
        self.assertAlmostEqual(float(joint),1,places=6)
        self.assertEqual(float(((1-q@a.T).min()+(1-q@b.T).min())/2),0)
