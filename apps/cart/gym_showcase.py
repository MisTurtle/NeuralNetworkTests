from apps.cart.ai_debug_environment import CartAiDebugEnvironment
from apps.cart.environment import CartEnvironment_V2
from apps.utils.gym_utils import Epsilon

env = CartEnvironment_V2()

model = env.create_model()
model.load_weights("models/sp_30000.h5")
debug_env = CartAiDebugEnvironment(env, model, Epsilon.constant(0.00))
debug_env.run()
