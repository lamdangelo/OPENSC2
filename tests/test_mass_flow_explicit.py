"""Unit tests of the explicit (mdot, p, T) Gauss-point builders
(hydraulics/mass_flow_equations.py) against the REAL helium property library.

States: 20 seeded random supercritical-helium states in 4-40 K, 3-10 bar,
Mach numbers |v|/c in 1e-4 ... 1e-1 (either sign), channel areas 1e-5 ...
1e-3 m^2, evaluated with CoolProp's direct HEOS flash (the tabulated
property path is switched off for the identity-level tolerances; a
separate reporting test measures the identity residuals of the tabulated
path).

Checked:
* eigenvalues of K~ equal {v-c, v, v+c} (rel. 1e-10), trace(K~) = 3v;
* K~ equals J^-1 K J built from the same properties, K the velocity-form
  flux Jacobian of the production builder build_amat;
* the mdot row of S~ W is exactly (2 f |mdot| / (rho A D_h)) mdot and does
  not depend on the heat-exchange entries q_p / q_T;
* kappa_T q_p - beta q_T = 0 on the assembled source block (fluid-solid and
  fluid-fluid heat exchange assembled with the production builders), the
  mdot row carrying no temperature column and the interface momentum +-K2;
* kappa_T phi - beta/(rho c_v) = 0 for the property wrapper.
"""

from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest

import interfaces.coolprop_interface as cpi
from hydraulics.hydraulic_flags import FluidType
from hydraulics.hydraulics import compute_gruneisen
from hydraulics.mass_flow_equations import (
    build_amat_mass_flow,
    build_kmat_fluid_mass_flow,
    build_smat_fluid_energy_mass_flow,
    build_smat_fluid_interface_momentum_mass_flow,
    build_smat_fluid_momentum_mass_flow,
    check_source_consistency,
    thermodynamic_coefficients,
)
from physical_fields.physical_field import FieldContainer, GridLocation
from thermal.energy_equation import (
    build_smat_fluid_interface_energy,
    build_smat_fluid_solid_interface,
)
from utility_functions.step_matrix_construction import build_amat

NUMBER_OF_STATES = 20
SEED = 20260908

PROPERTY_ALIASES = dict(
    isobaric_expansion_coefficient="isobaric_expansion_coefficient",
    isothermal_compressibility="isothermal_compressibility",
    total_density="Dmass",
    total_dynamic_viscosity="viscosity",
    total_enthalpy="Hmass",
    total_isobaric_specific_heat="Cpmass",
    total_isochoric_specific_heat="Cvmass",
    total_speed_of_sound="speed_of_sound",
)

EquationIndex = namedtuple("EquationIndex", ("velocity", "pressure", "temperature"))
Interface = namedtuple("Interface", ("interf_name", "comp_1", "comp_2"))


# --------------------------------------------------------------------------
# State and stub construction
# --------------------------------------------------------------------------


@pytest.fixture
def direct_eos(monkeypatch):
    """Evaluate properties with the direct HEOS flash (identities hold to
    roundoff) instead of the bicubic property tables (~1e-6)."""
    monkeypatch.setattr(cpi, "USE_TABULAR_PROPERTIES", False)


def sample_states(count=NUMBER_OF_STATES, seed=SEED) -> dict:
    """Random helium states with properties from the real library."""
    rng = np.random.default_rng(seed)
    temperature = rng.uniform(4.0, 40.0, count)
    pressure = rng.uniform(3.0e5, 10.0e5, count)
    properties = cpi.compute_properties(
        FluidType.HELIUM, PROPERTY_ALIASES, temperature, pressure
    )
    mach = 10.0 ** rng.uniform(-4.0, -1.0, count)
    sign = np.where(rng.uniform(size=count) < 0.5, -1.0, 1.0)
    velocity = sign * mach * properties["total_speed_of_sound"]
    cross_section = 10.0 ** rng.uniform(-5.0, -3.0, count)
    hydraulic_diameter = 10.0 ** rng.uniform(-3.5, -2.0, count)
    friction_factor = 10.0 ** rng.uniform(-2.5, -1.0, count)
    state = dict(
        temperature=temperature,
        pressure=pressure,
        velocity=velocity,
        cross_section=cross_section,
        hydraulic_diameter=hydraulic_diameter,
        friction_factor=friction_factor,
        **properties,
    )
    state["mass_flow_rate"] = (
        state["total_density"] * cross_section * velocity
    )
    state["Gruneisen"] = compute_gruneisen(
        state["isobaric_expansion_coefficient"],
        state["isothermal_compressibility"],
        state["total_isochoric_specific_heat"],
        state["total_density"],
    )
    return state


FIELD_NAMES = (
    "temperature",
    "pressure",
    "velocity",
    "mass_flow_rate",
    "Gruneisen",
) + tuple(PROPERTY_ALIASES)


def make_fluid_stub(identifier: str, state: dict, cross_section: float,
                    hydraulic_diameter: float):
    """Minimal stand-in for a FluidComponent with the attributes the
    builders read: coolant.gauss_fields, coolant.node_fields.velocity,
    channel.inputs.{cross_section, hydraulic_diameter} and
    channel.friction_factors[False].total."""
    gauss_fields = FieldContainer.from_mapping(
        {name: np.array(state[name], dtype=float) for name in FIELD_NAMES},
        GridLocation.GAUSS,
    )
    count = gauss_fields.velocity.size
    node_velocity = np.concatenate(
        ([gauss_fields.velocity[0]], gauss_fields.velocity)
    )
    node_fields = FieldContainer.from_mapping(
        {"velocity": node_velocity}, GridLocation.NODE
    )
    return SimpleNamespace(
        identifier=identifier,
        coolant=SimpleNamespace(gauss_fields=gauss_fields, node_fields=node_fields),
        channel=SimpleNamespace(
            inputs=SimpleNamespace(
                cross_section=cross_section,
                hydraulic_diameter=hydraulic_diameter,
            ),
            friction_factors={
                False: SimpleNamespace(
                    total=np.array(state["friction_factor"], dtype=float)
                )
            },
        ),
    )


def single_channel_stub(state: dict, index: int):
    """One Gauss point (state index) as a one-element channel, so that every
    sampled state is exercised with its own area and diameter."""
    point = {name: np.atleast_1d(np.asarray(state[name]))[index : index + 1]
             for name in state}
    return make_fluid_stub(
        "CHAN_1",
        point,
        float(state["cross_section"][index]),
        float(state["hydraulic_diameter"][index]),
    )


def make_conductor_stub(fluids, solids, fluid_fluid, fluid_solid, count,
                        rng, lambda_v=0.8):
    """Two-channel + one-solid conductor stand-in with open fluid-fluid
    interface and fluid-solid contacts, carrying the K1/K2/K3, contact
    perimeters and heat transfer coefficients the builders read."""
    equation_index = {}
    number_of_fluids = len(fluids)
    for j, f_comp in enumerate(fluids):
        equation_index[f_comp.identifier] = EquationIndex(
            j, j + number_of_fluids, j + 2 * number_of_fluids
        )
    for l, s_comp in enumerate(solids):
        equation_index[s_comp.identifier] = 3 * number_of_fluids + l
    ndf = 3 * number_of_fluids + len(solids)

    interf_peri = {
        "ch_ch": {"Open": {"Gauss": {}}, "Close": {"Gauss": {}}},
        "ch_sol": {"Gauss": {}},
    }
    htc = {"ch_ch": {"Open": {}, "Close": {}}, "ch_sol": {}}
    K1, K2, K3 = {}, {}, {}
    for interface in fluid_fluid:
        name = interface.interf_name
        interf_peri["ch_ch"]["Open"]["Gauss"][name] = rng.uniform(1e-3, 1e-2, count)
        interf_peri["ch_ch"]["Close"]["Gauss"][name] = rng.uniform(1e-3, 1e-2, count)
        htc["ch_ch"]["Open"][name] = rng.uniform(1e2, 1e4, count)
        htc["ch_ch"]["Close"][name] = rng.uniform(1e2, 1e4, count)
        # Donor-side transport coefficients as thermal_transport builds
        # them: pick the donor at random per Gauss point.
        donor = np.where(
            rng.uniform(size=count) < 0.5,
            interface.comp_1.coolant.gauss_fields.velocity,
            interface.comp_2.coolant.gauss_fields.velocity,
        )
        donor_enthalpy = np.where(
            rng.uniform(size=count) < 0.5,
            interface.comp_1.coolant.gauss_fields.total_enthalpy,
            interface.comp_2.coolant.gauss_fields.total_enthalpy,
        )
        K1[name] = rng.uniform(1e-5, 1e-3, count)
        K2[name] = K1[name] * lambda_v * donor
        K3[name] = K1[name] * (donor_enthalpy + 0.5 * (lambda_v * donor) ** 2)
    for interface in fluid_solid:
        name = interface.interf_name
        interf_peri["ch_sol"]["Gauss"][name] = rng.uniform(1e-2, 1e-1, count)
        htc["ch_sol"][name] = rng.uniform(1e2, 1e4, count)

    return SimpleNamespace(
        equation_index=equation_index,
        inventory=SimpleNamespace(
            fluids=SimpleNamespace(collection=list(fluids)),
            solids=SimpleNamespace(collection=list(solids)),
        ),
        interface=SimpleNamespace(
            fluid_fluid=list(fluid_fluid), fluid_solid=list(fluid_solid)
        ),
        dict_interf_peri=interf_peri,
        gauss_fields=SimpleNamespace(HTC=htc, K1=K1, K2=K2, K3=K3),
        mesh=SimpleNamespace(element_lengths=rng.uniform(0.01, 0.5, count)),
        ndf=ndf,
    )


def two_channel_system(seed=SEED):
    """Two channels (independent random states, shared point count) with an
    open interface, both wetting one solid; returns (conductor, S~)."""
    rng = np.random.default_rng(seed + 1)
    state_1 = sample_states(seed=seed)
    state_2 = sample_states(seed=seed + 7)
    count = state_1["velocity"].size
    fluid_1 = make_fluid_stub("CHAN_1", state_1, 2.0e-4, 5.0e-3)
    fluid_2 = make_fluid_stub("CHAN_2", state_2, 1.0e-4, 3.0e-3)
    solid = SimpleNamespace(identifier="STR_1")
    fluid_fluid = [Interface("CHAN_1_CHAN_2", fluid_1, fluid_2)]
    fluid_solid = [
        Interface("CHAN_1_STR_1", fluid_1, solid),
        Interface("CHAN_2_STR_1", fluid_2, solid),
    ]
    conductor = make_conductor_stub(
        [fluid_1, fluid_2], [solid], fluid_fluid, fluid_solid, count, rng
    )
    return conductor, assemble_source_block(conductor)


def assemble_source_block(conductor) -> np.ndarray:
    """S~ assembled exactly in the production order of the explicit path."""
    count = conductor.mesh.element_lengths.size
    matrix = np.zeros((count, conductor.ndf, conductor.ndf))
    for f_comp in conductor.inventory.fluids.collection:
        eq_idx = conductor.equation_index[f_comp.identifier]
        matrix = build_smat_fluid_momentum_mass_flow(matrix, f_comp, eq_idx)
        matrix = build_smat_fluid_energy_mass_flow(matrix, f_comp, eq_idx)
    matrix = build_smat_fluid_interface_momentum_mass_flow(matrix, conductor)
    matrix = build_smat_fluid_interface_energy(matrix, conductor)
    matrix = build_smat_fluid_solid_interface(matrix, conductor)
    return matrix


def jacobian_of_change_of_variables(state: dict, index: int) -> np.ndarray:
    """J = dW/dW' with W = (v, p, T), W' = (mdot, p, T)."""
    rho_area = state["total_density"][index] * state["cross_section"][index]
    velocity = state["velocity"][index]
    return np.array(
        [
            [
                1.0 / rho_area,
                -velocity * state["isothermal_compressibility"][index],
                velocity * state["isobaric_expansion_coefficient"][index],
            ],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


# --------------------------------------------------------------------------
# Flux Jacobian K~
# --------------------------------------------------------------------------


def explicit_flux_jacobians(state) -> np.ndarray:
    count = state["velocity"].size
    blocks = np.zeros((count, 3, 3))
    for index in range(count):
        f_comp = single_channel_stub(state, index)
        blocks[index] = build_amat_mass_flow(
            np.zeros((1, 3, 3)), f_comp, EquationIndex(0, 1, 2)
        )[0]
    return blocks


def test_flux_jacobian_spectrum_and_trace(direct_eos):
    state = sample_states()
    blocks = explicit_flux_jacobians(state)
    for index in range(NUMBER_OF_STATES):
        velocity = state["velocity"][index]
        sound = state["total_speed_of_sound"][index]
        analytic = np.sort([velocity - sound, velocity, velocity + sound])
        eigenvalues = np.linalg.eigvals(blocks[index])
        assert np.abs(eigenvalues.imag).max() < 1e-10 * sound
        deviation = np.abs(np.sort(eigenvalues.real) - analytic).max()
        assert deviation < 1e-10 * (abs(velocity) + sound), (
            f"state {index}: eigenvalues {np.sort(eigenvalues.real)} vs "
            f"{analytic} (T = {state['temperature'][index]:.2f} K, "
            f"p = {state['pressure'][index]:.3e} Pa)"
        )
        trace = np.trace(blocks[index])
        assert abs(trace - 3.0 * velocity) < 1e-12 * (abs(velocity) + sound), (
            f"state {index}: trace {trace} != 3 v = {3 * velocity}"
        )


def test_flux_jacobian_equals_conjugated_velocity_form(direct_eos):
    """K~ == J^-1 K J with K from the production velocity builder."""
    state = sample_states()
    explicit = explicit_flux_jacobians(state)
    for index in range(NUMBER_OF_STATES):
        f_comp = single_channel_stub(state, index)
        velocity_form = build_amat(
            np.zeros((1, 3, 3)), f_comp, EquationIndex(0, 1, 2)
        )[0]
        jacobian = jacobian_of_change_of_variables(state, index)
        reference = np.linalg.solve(jacobian, velocity_form @ jacobian)
        scale = np.abs(reference).max(axis=1, keepdims=True)
        deviation = (np.abs(explicit[index] - reference) / scale).max()
        assert deviation < 1e-10, (
            f"state {index}: K~ deviates from J^-1 K J by {deviation:.3e}\n"
            f"explicit:\n{explicit[index]}\nreference:\n{reference}"
        )


def test_stabilisation_block_keeps_velocity_structure(direct_eos):
    """Diagonal artificial diffusion only, dz*(u + c)/2 on the mdot row and
    dz*u/2 on the p and T rows (the velocity-formulation structure)."""
    conductor, _ = two_channel_system()
    f_comp = conductor.inventory.fluids.collection[0]
    eq_idx = conductor.equation_index[f_comp.identifier]
    upweqt = np.ones(conductor.ndf)
    block = build_kmat_fluid_mass_flow(
        np.zeros((conductor.mesh.element_lengths.size, conductor.ndf, conductor.ndf)),
        upweqt,
        f_comp,
        conductor,
    )
    fields = f_comp.coolant.gauss_fields
    dz = conductor.mesh.element_lengths
    node_speed = np.abs(f_comp.coolant.node_fields.velocity)
    element_speed = np.maximum(node_speed[:-1], node_speed[1:])
    padded = np.pad(element_speed, 1, mode="edge")
    speed = np.maximum(np.maximum(padded[:-2], padded[1:-1]), padded[2:])
    assert np.array_equal(
        block[:, eq_idx.velocity, eq_idx.velocity],
        dz * (speed + fields.total_speed_of_sound) / 2.0,
    )
    assert np.array_equal(block[:, eq_idx.pressure, eq_idx.pressure], dz * speed / 2.0)
    assert np.array_equal(
        block[:, eq_idx.temperature, eq_idx.temperature], dz * speed / 2.0
    )
    off_diagonal = block.copy()
    for slot in eq_idx:
        off_diagonal[:, slot, slot] = 0.0
    assert not off_diagonal.any()


# --------------------------------------------------------------------------
# Source block S~
# --------------------------------------------------------------------------


def test_mdot_row_is_pure_friction_single_channel(direct_eos):
    """(S~ W)_mdot == (2 f |mdot| / (rho A D_h)) mdot exactly, whatever the
    heat-exchange entries of the p and T rows are."""
    state = sample_states()
    rng = np.random.default_rng(SEED + 3)
    for index in range(NUMBER_OF_STATES):
        f_comp = single_channel_stub(state, index)
        solid = SimpleNamespace(identifier="STR_1")
        conductor = make_conductor_stub(
            [f_comp], [solid], [], [Interface("CHAN_1_STR_1", f_comp, solid)],
            1, rng,
        )
        block = assemble_source_block(conductor)[0]
        mdot = state["mass_flow_rate"][index]
        rho_area = state["total_density"][index] * state["cross_section"][index]
        expected = (
            2.0
            * state["friction_factor"][index]
            * abs(mdot)
            / (rho_area * state["hydraulic_diameter"][index])
        ) * mdot
        # Only the (mdot, mdot) entry of the first row is nonzero...
        assert block[0, 0] != 0.0
        assert not block[0, 1:].any()
        # ... so the row product is exactly the friction term.
        unknowns = np.array([mdot, state["pressure"][index],
                             state["temperature"][index],
                             state["temperature"][index] + 1.0])
        assert (block @ unknowns)[0] == block[0, 0] * mdot
        assert abs(block[0, 0] * mdot - expected) <= 4 * np.finfo(float).eps * abs(expected)
        # And the p / T rows do carry the heat exchange (the test is not
        # vacuous): the solid temperature column is nonzero there.
        assert block[1, 3] != 0.0 and block[2, 3] != 0.0


def test_mdot_row_independent_of_heat_exchange(direct_eos):
    """Doubling every heat transfer coefficient leaves the mdot row
    bit-identical (no dependence on q_p / q_T), on the two-channel system."""
    conductor, block = two_channel_system()
    for name in conductor.gauss_fields.HTC["ch_sol"]:
        conductor.gauss_fields.HTC["ch_sol"][name] *= 2.0
    for kind in ("Open", "Close"):
        for name in conductor.gauss_fields.HTC["ch_ch"][kind]:
            conductor.gauss_fields.HTC["ch_ch"][kind][name] *= 2.0
    doubled = assemble_source_block(conductor)
    for f_comp in conductor.inventory.fluids.collection:
        row = conductor.equation_index[f_comp.identifier].velocity
        assert np.array_equal(block[:, row, :], doubled[:, row, :])
    # The pressure and temperature rows did change.
    assert not np.array_equal(block, doubled)


def test_mdot_row_interface_momentum(direct_eos):
    """At an open interface the mdot row of channel j carries exactly +K2 on
    its own pressure column and -K2 on the partner's, nothing else besides
    friction."""
    conductor, block = two_channel_system()
    interface = conductor.interface.fluid_fluid[0]
    K2 = conductor.gauss_fields.K2[interface.interf_name]
    for comp_1, comp_2 in (
        (interface.comp_1, interface.comp_2),
        (interface.comp_2, interface.comp_1),
    ):
        j = conductor.equation_index[comp_1.identifier]
        k = conductor.equation_index[comp_2.identifier]
        row = block[:, j.velocity, :]
        assert np.array_equal(row[:, j.pressure], K2)
        assert np.array_equal(row[:, k.pressure], -K2)
        nonzero_columns = {j.velocity, j.pressure, k.pressure}
        for column in range(conductor.ndf):
            if column not in nonzero_columns:
                assert not row[:, column].any(), (
                    f"{comp_1.identifier}: mdot row has column {column}"
                )


def test_source_consistency_assertion_passes(direct_eos):
    """kappa_T q_p - beta q_T = 0 on every temperature column, mdot row free
    of temperature columns, interface momentum -K2, partner-pressure
    residual -K1/(rho A): the debug check passes and reports the residuals."""
    conductor, block = two_channel_system()
    residuals = check_source_consistency(block, conductor)
    print(f"\nsource-consistency residuals (direct EOS): {residuals}")
    assert residuals["temperature"] < 1e-13
    assert residuals["mdot_row"] < 1e-13
    # The partner-pressure relation rests on Reech + Mayer, exact with the
    # direct HEOS flash.
    assert residuals["pressure"] < 1e-10


def test_source_consistency_assertion_detects_inconsistent_rows(direct_eos):
    """Perturbing the temperature-row heat exchange alone must trip the
    assertion (guards against the check being vacuous)."""
    conductor, block = two_channel_system()
    f_comp = conductor.inventory.fluids.collection[0]
    j = conductor.equation_index[f_comp.identifier]
    solid_column = conductor.equation_index["STR_1"]
    block[:, j.temperature, solid_column] *= 1.0 + 1e-6
    with pytest.raises(AssertionError, match="kappa_T q_p - beta q_T"):
        check_source_consistency(block, conductor)


# --------------------------------------------------------------------------
# Property wrapper identities
# --------------------------------------------------------------------------


def test_gruneisen_identity(direct_eos):
    """kappa_T phi - beta/(rho c_v) = 0 at every sampled state."""
    state = sample_states()
    fields = FieldContainer.from_mapping(
        {name: state[name] for name in FIELD_NAMES}, GridLocation.GAUSS
    )
    _, gamma, gruneisen = thermodynamic_coefficients(fields, 1.0)
    residual = state["isothermal_compressibility"] * gruneisen - state[
        "isobaric_expansion_coefficient"
    ] / (state["total_density"] * state["total_isochoric_specific_heat"])
    scale = state["isothermal_compressibility"] * gruneisen
    assert np.all(np.abs(residual) <= 4 * np.finfo(float).eps * np.abs(scale))
    assert np.allclose(
        gamma,
        state["total_isobaric_specific_heat"] / state["total_isochoric_specific_heat"],
        rtol=0.0, atol=0.0,
    )


def identity_residuals(state) -> dict:
    rho = state["total_density"]
    gamma = state["total_isobaric_specific_heat"] / state["total_isochoric_specific_heat"]
    reech = rho * state["total_speed_of_sound"] ** 2 * state["isothermal_compressibility"]
    mayer = 1.0 + state["Gruneisen"] * state["isobaric_expansion_coefficient"] * state["temperature"]
    return {
        "Reech rho c^2 kappa_T / gamma - 1": float(np.abs(reech / gamma - 1.0).max()),
        "Mayer (1 + phi beta T) / gamma - 1": float(np.abs(mayer / gamma - 1.0).max()),
    }


def test_thermodynamic_identities_direct_eos(direct_eos):
    """Reech and Mayer hold to roundoff with the direct HEOS flash: this is
    what makes the explicit coefficient listing (2v, -(gamma-1)v, gamma v)
    identical to the conjugated velocity form."""
    residuals = identity_residuals(sample_states())
    print(f"\nidentity residuals (direct EOS): {residuals}")
    for name, value in residuals.items():
        assert value < 1e-10, f"{name}: {value:.3e}"


def test_thermodynamic_identities_tabulated_properties_report():
    """REPORTING test: the identity residuals of the production property
    path (bicubic tables, USE_TABULAR_PROPERTIES). Not an identity-level
    assertion; the loose bound only flags gross table breakage. The
    measured level (~1e-6 .. 1e-4) is the level at which the explicit
    coefficients and the transform route differ in production runs."""
    assert cpi.USE_TABULAR_PROPERTIES
    residuals = identity_residuals(sample_states())
    print(f"\nidentity residuals (tabulated properties): {residuals}")
    for name, value in residuals.items():
        assert value < 1e-2, f"{name}: {value:.3e}"
