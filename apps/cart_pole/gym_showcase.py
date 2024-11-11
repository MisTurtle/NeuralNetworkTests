from apps.cart_pole.ai_debug_environment import CartPoleAiDebugEnvironment
from apps.cart_pole.environment import *
from apps.utils.gym_utils import Epsilon

env = CartPoleEnvironment_V3()

model = env.create_model()
model.load_weights("models/cartpole_2000.h5")
debug_env = CartPoleAiDebugEnvironment(env, model, Epsilon.constant(0))
debug_env.run()
