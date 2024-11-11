from apps.cart_pole.debug_environment import CartPoleDebugEnvironment
from apps.cart_pole.environment import *

sim = CartPoleDebugEnvironment(CartPoleEnvironment_V3())
sim.run()
