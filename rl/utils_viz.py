import numpy as np
import torch
import visdom
import matplotlib.pyplot as plt

global_vis = visdom.Visdom(port=8097, server='http://visiongpu20', use_incoming_socket=True)
CMAP = plt.get_cmap('plasma')

def tonumpy(tensor):
  if type(tensor) is list:
    return np.array(tensor)
  if type(tensor) is np.ndarray:
    return tensor
  if tensor.requires_grad:
    tensor = tensor.detach()
  if type(tensor) is torch.autograd.Variable:
    tensor = tensor.data
  if tensor.is_cuda:
    tensor = tensor.cpu()
  return tensor.detach().numpy()

def prepare_pointclouds_and_colors(coords, colors, default_color=(0,0,0)):
  if type(coords) is list:
    for k in range(len(coords)):

      assert len(coords) == len(colors)
      coords[k], colors[k] = prepare_single_pointcloud_and_colors(coords[k], colors[k], default_color)
    return coords, colors
  else:
    return prepare_single_pointcloud_and_colors(coords, colors, default_color)


def list_of_lists_into_single_list(list_of_lists):
  flat_list = [item for sublist in list_of_lists for item in sublist]
  return flat_list

def prepare_single_pointcloud_and_colors(coords, colors, default_color=(0,0,0)):
  coords = tonumpy(coords)
  if colors is None:
    colors = np.array(default_color)[:, None].repeat(coords.size / 3, 1).reshape(coords.shape)
  colors = tonumpy(colors)
  if colors.dtype == 'float32':
    if colors.max() > 1.0:
      colors = np.array(colors, dtype='uint8')
    else:
      colors = np.array(colors * 255.0, dtype='uint8')
  if type(coords) is list:
    for k in range(len(colors)):
      if colors[k] is None:
        colors[k] = np.ones(coords[k].shape)
    colors = np.concatenate(colors, axis=0)
    coords = np.concatenate(coords, axis=0)

  print(coords.shape, colors.shape)
  assert coords.shape == colors.shape
  if len(coords.shape) == 3:
    coords = coords.reshape((3, -1))
  if len(colors.shape) == 3:
    colors = colors.reshape((3, -1))
  assert len(coords.shape) == 2
  if coords.shape[0] == 3:
    coords = coords.transpose()
    colors = colors.transpose()
  return coords, colors

def show_pointcloud(original_coords, original_colors=None, history=None, title='none', win=None, env='main',
                    markersize=3, max_points=10000, valid_mask=None, labels=None, default_color=(0,0,0),
                    projection="orthographic", center=(0,0,0), up=(0,-1,0), eye=(0,0,-2),
                    display_grid=(True,True,True), axis_ranges=None):
  assert projection in ["perspective", "orthographic"]
  coords, colors = prepare_pointclouds_and_colors(original_coords, original_colors, default_color)
  if not type(coords) is list:
    coords = [coords]
    colors = [colors]
  if not valid_mask is None:
    if not type(valid_mask) is list:
      valid_mask = [valid_mask]
    assert len(valid_mask) == len(coords)
    for i in range(len(coords)):
      if valid_mask[i] is None:
        continue
      else:
        actual_valid_mask = np.array(valid_mask[i], dtype='bool').flatten()
        coords[i] = coords[i][actual_valid_mask]
        colors[i] = colors[i][actual_valid_mask]
  if not labels is None:
    if not type(labels) is list:
      labels = [labels]

    assert len(labels) == len(coords)
  for i in range(len(coords)):
    if max_points != -1 and coords[i].shape[0] > max_points:
      selected_positions = random.sample(range(coords[i].shape[0]), max_points)
      coords[i] = coords[i][selected_positions]
      colors[i] = colors[i][selected_positions]
      if not labels is None:
        labels[i] = [labels[i][k] for k in selected_positions]
      if not type(markersize) is int or type(markersize) is float:
        markersize[i] = [markersize[i][k] for k in selected_positions]
  # after this, we can compact everything into a single set of pointclouds. and do some more stuff for nicer visualization
  coords = np.concatenate(coords)
  colors = np.concatenate(colors)
  if not type(markersize) is int or type(markersize) is float:
    markersize = list_of_lists_into_single_list(markersize)
    assert len(coords) == len(markersize)
  if not labels is None:
    labels = list_of_lists_into_single_list(labels)
  if win is None:
    win = title
  plot_coords = coords
  from visdom import _markerColorCheck
  # we need to construct our own colors to override marker plotly options
  # and allow custom hover (to show real coords, and not the once used for visualization)
  visdom_colors = _markerColorCheck(colors, plot_coords, np.ones(len(plot_coords), dtype='uint8'), 1)
  # add the coordinates as hovertext
  hovertext = ['x:{:.2f}\ny:{:.2f}\nz:{:.2f}\n'.format(float(k[0]), float(k[1]), float(k[2])) for k in coords]
  if not labels is None:
    assert len(labels) == len(hovertext)
    hovertext = [hovertext[k] + ' {}'.format(labels[k]) for k in range(len(hovertext))]

  # to see all the options interactively, click on edit plot on visdom->json->tree
  # traceopts are used in line 1610 of visdom.__intit__.py
  # data.update(trace_opts[trace_name])
  # for layout options look at _opts2layout

  camera = {'up':{
              'x': str(up[0]),
              'y': str(up[1]),
              'z': str(up[2]),
            },
            'eye':{
              'x': str(eye[0]),
              'y': str(eye[1]),
              'z': str(eye[2]),
            },
            'center':{
              'x': str(center[0]),
              'y': str(center[1]),
              'z': str(center[2]),
            },
            'projection': {
              'type': projection
            }
          }

  global_vis.scatter(plot_coords, env=env, win=win,
              opts={'webgl': True,
                    'title': title,
                    'name': 'scatter',
                    'layoutopts': {
                      'plotly':{
                        'scene': {
                          'aspectmode': 'data',
                          'camera': camera,
                          'xaxis': {
                            'tickfont':{
                              'size': 14
                            },
                            'autorange': axis_ranges is None,
                            'range': [str(axis_ranges['min_x']), str(axis_ranges['max_x'])] if not axis_ranges is None else [-1,-1],
                            'showgrid': display_grid[0],
                            'showticklabels': display_grid[0],
                            'zeroline': display_grid[0],
                            'title': {
                                  'text':'x' if display_grid[0] else '',
                                  'font':{
                                    'size':20
                                    }
                                  }
                          },
                          'yaxis': {
                            'tickfont':{
                              'size': 14
                            },
                            'autorange': axis_ranges is None,
                            'range': [str(axis_ranges['min_y']), str(axis_ranges['max_y'])] if not axis_ranges is None else [-1, -1],
                            'showgrid': display_grid[1],
                            'showticklabels': display_grid[1],
                            'zeroline': display_grid[1],
                            'title': {
                                  'text':'y' if display_grid[1] else '',
                                  'font':{
                                    'size':20
                                    }
                                  }
                          },
                          'zaxis': {
                            'tickfont':{
                              'size': 14
                            },
                            'autorange': axis_ranges is None,
                            'range': [str(axis_ranges['min_z']), str(axis_ranges['max_z'])] if not axis_ranges is None else [-1, -1],
                            'showgrid': display_grid[2],
                            'showticklabels': display_grid[2],
                            'zeroline': display_grid[2],
                            'title': {
                                  'text':'z' if display_grid[2] else '',
                                  'font':{
                                    'size':20
                                    }
                                  }
                          }
                        }
                      }
                    },
                    'traceopts': {
                      'plotly':{
                        '1': {
                          #custom ops
                          # https://plot.ly/python/reference/#scattergl-transforms
                          'hoverlabel':{
                            'bgcolor': '#000000'
                          },
                          'hoverinfo': 'text',
                          'hovertext': hovertext,
                          'marker': {
                            'sizeref': 1,
                            'size': markersize,
                            'symbol': 'dot',
                            'color': visdom_colors[1],
                            'line': {
                                'color': '#000000',
                                'width': 0,
                            }
                          }
                        },
                      }
                    }
                  })

  # if history is not None:
  #   time = list(range(history.shape[1]))
  #   global_vis.scatter(history.transpose(),
  #                      env=env, win=win,
  #                      opts= {
  #                        'plotly':{
  #                         '1': {
  #                           #custom ops
  #                           # https://plot.ly/python/reference/#scattergl-transforms
  #                           'hoverlabel':{
  #                             'bgcolor': '#000000'
  #                           },
  #                           'hoverinfo': 'text',
  #                           'hovertext': hovertext,
  #                           'marker': {
  #                             'sizeref': 1,
  #                             'size': markersize/2.,
  #                             'symbol': 'dot',
  #                             'color_value':time,
  #                             'line': {
  #                                 'color': '#000100',
  #                                 'width': 0.3,
  #                             }
  #                           }
  #                         },
  #
  #                      }})
  return

def imshow_vis(im, title=None, win=None, env=None, vis=None):
  if vis is None:
    vis = global_vis
  opts = dict()
  win, title, vis = visdom_default_window_title_and_vis(win, title, vis)

  opts['title'] = title
  if im.dtype is np.uint8:
    im = im/255.0
  vis.image(im, win=win, opts=opts, env=env)


def visdom_default_window_title_and_vis(win, title, vis):
  if win is None and title is None:
    win = title = 'None'
  elif win is None:
    win = str(title)
  elif title is None:
    title = str(win)
  if vis is None:
    vis = global_vis
    vis = global_vis
  return win, title, vis

def convert_coords(coords):
  coords = [np.array(c)[:, None] for c in coords]
  coords = np.concatenate(coords, 1)[[0, 2, 1], :]
  coords[2, :] = 0.
  return coords

def plot_graph(graph, visible_nodes, env, target_class=['microwave'], history_locations=None):
    nodes = [node for node in graph['nodes'] if node['class_name'] not in ['ceiling', 'doorjamb', 'door', 'wall', 'floor']]

    coords = convert_coords([node['bounding_box']['center'] for node in nodes])
    visible_nodes_indices = np.array([i for i, node in enumerate(nodes) if node['id'] in visible_nodes])
    target_node_indices = np.array([i for i, node in enumerate(nodes) if node['class_name'] in target_class])

    character_indx = np.array([i for i, node in enumerate(nodes) if node['class_name'] == 'character'])

    labels = [node['class_name'] for node in nodes]
    # 3xN
    color = [np.array([0,0,255])[:, None] for _ in range(len(labels))]
    color = np.concatenate(color, 1)
    color[:, visible_nodes_indices] = np.array([255, 0, 0])[:, None]
    color[:, target_node_indices] = np.array([0, 255, 0])[:, None]
    color[:, character_indx] = np.array([120, 155, 0])[:, None]

    history_coords = convert_coords(history_locations)

    labels = [labels]

    labels.append(['Trace']*len(history_locations))
    coords = [coords, history_coords]
    color_hist = np.concatenate([np.array(CMAP(it*1./20))[:3, None]*255. for it in range(len(history_locations))], 1).astype(np.uint8)
    color = [color, color_hist]
    print("SHAPE", color_hist.shape, len(history_locations))
    markersize = [[15]*coords[0].shape[1], [8]*coords[1].shape[1]]
    #print(history.shape, coords.shape)
    show_pointcloud(coords, color, labels=labels, markersize=markersize, env=env)
    #self.vis.scatter(X=all_coords[:,0], Y=all_coords[:,1], opts={'textlabels': class_names})

def show_image(img, env):
  imshow_vis(img, env=env)
# if __name__ == '__main__':
#   coords = np.ones((3,100,100))
#   colors = np.ones((3,100,100), dtype='uint8')*255
#   show_pointcloud(coords, colors, labels)
#   imshow_vis(colors, env='main')