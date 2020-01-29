# Multi-agent-goal-inference

### VirtualHome Unity Code
* https://gitlab.com/kkhra/SyntheticVideos
* Description: VirtualHome source code (video and frame)
* RL: multiagent branch

### VirtualHome Python Interface
* https://github.com/xavierpuigf/virtualhome
* Description: API to run VirtualHome, a simulator to generate videos of human activities

```
object placing: resources/object_script_placing.json
object states: resources/object_states.json
```


### VirtualHome Python Code
* https://github.com/andrewliao11/vh_mdp
* Description: VirtualHome source code. Generating graph for each action (frame only)


### PDDL
* https://github.com/xavierpuigf/PDDLMultiAgent
* Description: generate plans


---
### Multi-agent Models
* https://docs.google.com/document/d/1oGXOOpoHNLnyUD4bvZHb0KsHul_ocpBK6BmpczBVLXU/edit  <br/> 
  Google Doc Introduction

* https://github.com/xavierpuigf/vh_multiagent_models  <br/> 
  This repo is for training models for Alice and Bob
  
  * **Install** 
    ```
    install vh_mdp: 
    git clone https://github.com/xavierpuigf/virtualhome.git -b virtualhome_pkg
    cd virtualhome_pkg
    pip install .

    git clone https://github.com/andrewliao11/vh_mdp.git -b search
    cd vh_mdp
    pip install .
    ```

  * **Models**  <br/> 
    1. **unified_agent branch: (Alice), RL, BC, Search**
       * trainer.py: BC
       * MCTS_mian.py: Search
       * single_agent.py: RL

    2. **belief branch: (BoB)**
  


### Challenge
your_agent_id = get_your_agent_id()
system_agent_action = get_system_agent_action()
all_agent_id = get_all_agent_id()
























