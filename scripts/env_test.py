import numpy as np
from gymnasium import make

from lib.wrappers.bipedal_walker.test import Test_Wrapper

SEED = 42

def main():
    
    env = make("BipedalWalker-v3", render_mode="human")
    wrap_env = Test_Wrapper(env, plotting=True)
    wrap_env.reset(seed=SEED)
    wrap_env.action_space.seed(SEED)
    
    
    while(1):    
        assert wrap_env.action_space.shape is not None
        
        # random agent
        action = wrap_env.action_space.sample()
        wrap_env.step(action)
        print("=== Testing with action: ", action)
        # print()
        # print();
    
if __name__ == "__main__":
    main()
