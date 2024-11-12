from apps.cart.debug_environment import CartDebugEnvironment
from apps.cart.environment import CartEnvironment_V2

sim = CartDebugEnvironment(CartEnvironment_V2())
sim.run()
