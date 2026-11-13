"""Inactive specification registrations, retained for reference."""

import jax.numpy as jnp
from dynamicmatching.helpers import (
    masksM, masksMS, masksKMS, tauMcal, tauMScal, tauMStrend, tauMScaltrend,
    tauMStri, tauKMS, scaleMcal, scaleMScaltrend,
)

tpl = lambda a: (a, a)
scale1 = lambda d: lambda _, dev: tpl(jnp.ones(d, dtype=jnp.float32))

specs = { "Mcal1":
            ("M", 2, masksM, tauMcal, scale1(2), 2, None,
             range(1999, 2021), False),
          "Mcal":
            ("M", 2, masksM, tauMcal, scaleMcal, 2+1, None,
             range(1999, 2021), False),
          "MStrend":
            ("MS", 3, masksMS, tauMStrend, scale1(3), 14, None,
             range(1999, 2021), False),
          "MScal1":
            ("MS", 3, masksMS, tauMScal, scale1(3), 5, None,
             range(1999, 2021), False),
          "MScalold":
            # Eight parameters; three free scales. Checkpoint alias: MScal.
            # HF revision: ba7727a7e639833895967f9fa4241b385bbcde91.
            ("MS", 3, masksMS, tauMScal,
             lambda par, dev: tpl(jnp.exp(par[5:])), 5+3, None,
             range(1999, 2021), False),
          "MScaltrend2":
            ("MS", 3, masksMS, tauMScaltrend, scaleMScaltrend, 10+2, None,
             range(1999, 2021), False),
          "MStri":
            ("MS", 3, masksMS, tauMStri, scale1(3), 8, None,
             range(1999, 2021), False),
          "KMS":
            ("KMS", 8, masksKMS, tauKMS, scale1(8), 12, None,
             range(1999, 2021), False),
          "KMSclosed":
            ("KMS", 8, masksKMS, tauKMS, scale1(8), 12, None,
             range(1999, 2021), False)
         }
