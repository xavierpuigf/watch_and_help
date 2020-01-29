### Functions

* your_agent_id = get_your_agent_id()
* system_agent_action = get_system_agent_action()
* all_agent_id = get_all_agent_id()

* system_agent_observations = get_system_agent_observations(modality)
* my_agent_observations = get_my_agent_observations(modality)
  * modality is a list of 'rgb_image', 'seg_class', 'seg_inst', 'depth', 'symbolic_state'



### Notes implementation
To get the cameras:

```
num_cameras = comm.camera_count()
rgb_img = camera_image([num_cameras-1])
```






















