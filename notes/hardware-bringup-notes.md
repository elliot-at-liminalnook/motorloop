# Hardware Bring-Up Notes

Early cautions collected before first power-up, moved here from the project
README. See also `hardware-photo-inventory.md` (component identification) and
`open-questions.md` (Q7: ZONRI board measurements).

- Verify the exact Gowin package printed on the FPGA module before finalizing
  constraints. Sipeed documentation found online for Tang Primer 25K
  references a GW5A-LV25MG121-family part, while our project notes mention
  GW5A-LV25PG138C1/I0.
- TXB0108 is intended for push-pull interfaces. Treat I2C through TXB0108 as
  suspect; prefer AS5600 PWM output with TXB0108, or use an I2C-suitable
  translator such as TXS/LSF/BSS138-style shifting for SDA/SCL.
- Treat TI's DRV830x-HC-C2-KIT docs as the best authoritative
  schematic/pinout reference for the ZONRI board, but do not transfer TI EVM
  current or thermal ratings to the ZONRI board without checking its actual
  MOSFETs, shunts, copper, cooling, and connectors.
- Before first power: current-limited bench supply, continuity checks, common
  ground checks, no motor attached for first logic tests, and scope/logic
  analyzer on PWM and EN_GATE.

## OpenModelica study trail

Recommended first walk through the OpenModelica examples (see
`openmodelica-example-tour.md` for the full tour):

- `HelloWorld.mo` and `SimpleIntegrator.mo`: smallest continuous-time state
  examples.
- `BouncingBall.mo`: hybrid event behavior with `when`, `pre`, and `reinit`.
- `dcmotor.mo`: first useful electrical/mechanical component connection
  example — the natural starting point for the oracle plant models.
- `BouncingBall.mos` in the FMI tests: export/import/simulate an FMU.
- `DualMassOscillator.mos`: compose two FMUs with OMSimulator connections.
- `testSynchronousFMU_02.mos`: clocked/sampled behavior exported as an FMU.

The FMU examples belong to the optional FMI learning track rather than the
verification critical path — the controller is verified in the lockstep
Verilator bench instead (see `architecture.md`).
