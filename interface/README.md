### Functions
```
your_agent_id = get_your_agent_id()
system_agent_action = get_system_agent_action()
all_agent_id = get_all_agent_id()
observation = get_observations(agent_id, image=False)
```


### Notes implementation
To get the cameras:

```
num_cameras = comm.camera_count()
rgb_img = camera_image([num_cameras-1])
```






















