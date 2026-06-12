within ;

package BldcCosimTestbench
  package Plants
    model OnePhaseElectromechanicalPlant
      parameter Real R(unit = "Ohm") = 0.5
        "Phase resistance placeholder";
      parameter Real L(unit = "H") = 0.001
        "Phase inductance placeholder";
      parameter Real Ke(unit = "V.s/rad") = 0.02
        "Back-EMF constant placeholder";
      parameter Real Kt(unit = "N.m/A") = 0.02
        "Torque constant placeholder";
      parameter Real J(unit = "kg.m2") = 2e-4
        "Rotor/load inertia placeholder";
      parameter Real B(unit = "N.m.s/rad") = 1e-4
        "Viscous damping placeholder";

      parameter Real i0(unit = "A") = 0.0
        "Initial winding current";
      parameter Real omega0(unit = "rad/s") = 0.0
        "Initial angular velocity";
      parameter Real theta0(unit = "rad") = 0.0
        "Initial angular position";

      input Real duty(min = -1.0, max = 1.0) = 0.0
        "Signed averaged bridge command";
      input Real vBus(unit = "V") = 12.0
        "DC bus voltage used by the ideal averaged bridge";
      input Real loadTorque(unit = "N.m") = 0.0
        "Positive torque opposing positive rotation";

      output Real dutyLimited(min = -1.0, max = 1.0)
        "Command after ideal saturation";
      output Real phaseVoltage(unit = "V")
        "Applied phase voltage";
      output Real backEmf(unit = "V")
        "Speed-proportional back-EMF";
      output Real current(unit = "A")
        "Winding current";
      output Real electromagneticTorque(unit = "N.m")
        "Current-proportional electromagnetic torque";
      output Real omega(unit = "rad/s")
        "Angular velocity";
      output Real theta(unit = "rad")
        "Angular position";

    protected
      Real i(unit = "A", start = i0, fixed = true);
      Real w(unit = "rad/s", start = omega0, fixed = true);
      Real phi(unit = "rad", start = theta0, fixed = true);

    equation
      dutyLimited = noEvent(min(1.0, max(-1.0, duty)));
      phaseVoltage = dutyLimited * vBus;
      backEmf = Ke * w;

      L * der(i) = phaseVoltage - R * i - backEmf;
      electromagneticTorque = Kt * i;
      J * der(w) = electromagneticTorque - B * w - loadTorque;
      der(phi) = w;

      current = i;
      omega = w;
      theta = phi;
    end OnePhaseElectromechanicalPlant;

    model ThreePhaseAveragedOpenLoop
      "Averaged six-step open-loop ramp; mirrors the averaged mode of sim/cpp/src/three_phase_plant.cpp and sim/scripts/run_three_phase_reference.py. Oracle runs override every parameter from sim/config/params.toml."
      constant Real pi = 3.14159265358979323846264338327950288;

      parameter Real R(unit = "Ohm") = 0.5 "Phase resistance";
      parameter Real L(unit = "H") = 0.001 "Phase inductance";
      parameter Real Ke(unit = "V.s/rad") = 0.02 "Peak back-EMF constant";
      parameter Real J(unit = "kg.m2") = 2e-4 "Rotor/load inertia";
      parameter Real B(unit = "N.m.s/rad") = 1e-4 "Viscous damping";
      parameter Integer polePairs = 4;
      parameter Real blend = 0.0 "EMF shape: 0 sinusoid .. 1 trapezoid";
      parameter Real loadTorque(unit = "N.m") = 0.0;
      parameter Real vBus(unit = "V") = 12.0;
      parameter Real duty = 0.5;
      parameter Real fElecFinal(unit = "Hz") = 40.0;
      parameter Real rampTime(unit = "s") = 0.100;
      parameter Real iEps(unit = "A") = 1e-6 "Float-mode current threshold";

      Real i[3](each start = 0, each fixed = true) "Phase currents into motor";
      Real omega(start = 0, fixed = true);
      Real theta(start = 0, fixed = true);

      Real phaseE "Commanded electrical phase";
      Integer sector "Six-step sector 0..5";
      Integer hiPhase "1-based driven-high phase";
      Integer loPhase "1-based driven-low phase";
      Real shape[3];
      Real e[3] "Phase back-EMFs";
      Boolean connected[3];
      Real vt[3] "Terminal voltages";
      Real vn "Neutral voltage";
      Real torque;

    protected
      constant Integer hiTable[6] = {1, 1, 2, 2, 3, 3};
      constant Integer loTable[6] = {2, 3, 3, 1, 1, 2};

    equation
      phaseE = 2*pi*(if time < rampTime then
        0.5*fElecFinal*time*time/rampTime
      else
        0.5*fElecFinal*rampTime + fElecFinal*(time - rampTime));
      sector = mod(integer(floor(phaseE/(pi/3))), 6);
      hiPhase = hiTable[sector + 1];
      loPhase = loTable[sector + 1];

      for k in 1:3 loop
        shape[k] = (1 - blend)*sin(polePairs*theta - (k - 1)*2*pi/3)
          + blend*max(-1, min(1, 2*sin(polePairs*theta - (k - 1)*2*pi/3)));
        e[k] = Ke*shape[k]*omega;
        connected[k] = (k == hiPhase) or (k == loPhase) or abs(i[k]) > iEps;
        // Driven legs are imposed; the unselected leg freewheels through
        // ideal averaged clamps (0 / vBus) while |i| > iEps, else floats.
        vt[k] = if k == hiPhase then duty*vBus
          elseif k == loPhase then 0
          elseif i[k] > iEps then 0
          elseif i[k] < -iEps then vBus
          else vn + e[k];
        der(i[k]) = if connected[k] then
          (vt[k] - vn - e[k] - R*i[k])/L
        else
          0;
      end for;

      // Isolated neutral over connected legs (currents sum to ~0 there);
      // the six-step schedule always keeps the driven pair connected.
      vn = sum(if connected[k] then vt[k] - e[k] else 0 for k in 1:3)
        /max(sum(if connected[k] then 1 else 0 for k in 1:3), 2);

      torque = Ke*(shape[1]*i[1] + shape[2]*i[2] + shape[3]*i[3]);
      der(omega) = (torque - B*omega - loadTorque)/J;
      der(theta) = omega;
    end ThreePhaseAveragedOpenLoop;
  end Plants;
end BldcCosimTestbench;
