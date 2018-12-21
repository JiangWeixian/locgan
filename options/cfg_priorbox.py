v1 = {
    'feature_maps': [36, 18, 5],
    'min_dim': 288,
    'steps': [8, 16, 58],
    'min_sizes': [30, 60, 161],
    'max_sizes': [60, 111, 213],
    'aspect_ratios': [[2], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'masK_overlap': 0.3,
    'name': 'v1'
    }

v2 = {
    'feature_maps': [36, 18, 5],
    'min_dim': 288,
    'steps': [8, 16, 72],
    'min_sizes': [30, 60, 111],
    'max_sizes': [60, 111, 168],
    'aspect_ratios': [[2], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'masK_overlap': 0.3,
    'name': 'v1'
}

v3 = {
    'feature_maps': [36, 18, 9],
    'min_dim': 288,
    'steps': [8, 16, 32],
    'min_sizes': [30, 60, 111],
    'max_sizes': [60, 111, 162],
    'aspect_ratios': [[2], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'masK_overlap': 0.3,
    'name': 'v1'
}

v4 = {
    'feature_maps' : [38, 19, 10, 5, 3, 2],
    'min_dim' : 300,
    'steps' : [8, 16, 32, 64, 100, 150],
    'min_sizes' : [30, 60, 111, 162, 213, 264],
    'max_sizes' : [60, 111, 162, 213, 264, 315],
    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'masK_overlap': 0.3,
    'name' : 'v1'
}

v5 = {
    'feature_maps' : [75],
    'min_dim' : 300,
    'steps' : [4],
    'min_sizes' : [30],
    'max_sizes' : [60],
    'aspect_ratios' : [[2, 3, 4]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'masK_overlap': 0.3,
    'name' : 'v1'
}

# v6 = {
#     'feature_maps' : [64, 32, 16, 8, 4, 2],
#     'min_dim' : 512,
#     'steps' : [8, 16, 32, 64, 128, 256],
#     'min_sizes' : [21, 51, 133, 215, 296, 378],
#     'max_sizes' : [51, 133, 215, 296, 378, 460],
#     'aspect_ratios' : [[2, 1.6], [2, 3, 1.6], [2, 3, 1.6], [2, 3, 1.6], [2], [2]],
#     'variance' : [0.1, 0.2],
#     'clip' : True,
#     'masK_overlap': 0.3,
#     'name' : 'v1'
# }

v6 = {
    'feature_maps' : [41, 21, 11, 6, 3, 2],
    'min_dim' : 321,
    'steps' : [8, 16, 32, 64, 100, 150],
    'min_sizes' : [30, 60, 111, 162, 213, 264],
    'max_sizes' : [60, 111, 162, 213, 264, 315],
    'aspect_ratios' : [[2], [2, 3, 1.6], [2, 3, 1.6], [2, 3, 1.6], [2], [2]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'masK_overlap': 0.3,
    'name' : 'v1'
}