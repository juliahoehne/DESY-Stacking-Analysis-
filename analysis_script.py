#!/usr/bin/env python

""""
This is an example script for running the full analysis
for a source population and N events in the dataset
for n energy bins.

For the source population, we simulate the sources' positions
in (DEC, RA) by drawing from a uniform distribution
(in a chosen DEC band).

The dataset is simulated wrt the number of signal events,
namely the total number of injected signal events is what defines
the number of background events in the dataset.
The sources' neutrino fluxes follow a power law, so by
having the expected fluxes we can calculate the
expected number of signal neutrinos per energy bin per source,
assuming a power-law energy spectrum.
Adding Poisson noise gives the injected number of signal events
which we use to simulate the dataset. Including a scale factor,
we can scale up or down the number of expected signal around which 
the Poisson distribution is created for the injected signal.
The number of expected and injected background events
(ie astrophysical & atmospheric) is calculated from their
respective power-law energy spectra.

For the signal events, their positions are simulated by
first drawing from a Gaussian distribution and simulating
mock true positions, which we then rotate to match the
sources positions. For the background events, the
positions are uniformly drawn and within the same DEC band
as for the sources.

The signal and background llhs have only the spatial terms,
so that their ratio, which defines the TS value, is calculated per energy bin.
The signal spatial pdf is a Gaussian that depends on the 
opening angles between the events and source's positions,
while the background spatial pdf is a uniform distribution.
The TS is a function of ns, ie number of signal events,
which you minimize to find the ns value that best fits the
dataset, and calculate the TS for this best-fit ns.
The minimization is done separately per energy bin,
so the best-fit ns (TS) are summed together to give us the 
total/final best-fit ns (TS) for the full energy range.
You do this many times (ie trials) to find a TS distribution,
where each time a different dataset is created for the 
same source population.

For each trial, the expected & injected number of signal events,
as well as the total best-fit ns and TS are saved in a file. 
Trials can be run in one go or in batches, so that
the results from each batch are appended in the file.
In case of batch trials, the source population should be
first saved and then loaded for each batch in order to
ensure that all the trials are for the same sources. 
"""

import os, argparse, ast, pickle
import healpy as hp
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, List, Union, Callable

from scipy.optimize import minimize
from scipy.stats import norm

import matplotlib.pyplot as plt


class SimulateSources:
    """
    Simulate the source population.
    Assign DEC & RA for the sources, assuming that
    they are isotropically distributed in the sky.
    Choose the DEC to be in a band of +- 60deg,
    so that you can avoid having events near the Poles
    since there the background events are more
    difficult to estimate, but also in order to
    account for the fact you can't see all the
    sources in the sky

    Args:
        nsources (int): number of sources you want to simulate
        min_dec (float): minimum DEC (in deg). Default is -60
        max_dec (float): maximum DEC (in deg). Default is 60
    """

    def __init__(
        self, nsources: int, min_dec: float = -60.0, max_dec: float = 60.0
    ) -> None:

        self.nsrcs = nsources
        self.min_dec = min_dec
        self.max_dec = max_dec

        # create source population
        self.simulate_src_positions()

    def simulate_src_positions(self) -> None:
        """Simulate sources positions uniformly distributed in the sky.
        For RA you simulate random values between [0, 360],
        for DEC between the min & max sin(DEC), and convert both in rad
        """

        self.src_positions = [
            (
                np.deg2rad(np.random.uniform(self.min_dec, self.max_dec)),
                np.deg2rad(np.random.uniform(0, 360)),
            )
            for _ in range(self.nsrcs)
        ]


class SimulateDataset:
    """
    Simulate the dataset containing signal and background events.
    The number of expected events are calculated wrt their
    power-law distributions for each energy bin,
    so that the number of simulated events are drawn from
    a Poisson distribution around the expectation value.
    The total number of simulated events is given by the N parameter,
    so the simulated dataset consists of N events positions in (RA, DEC)
    for each energy bin.

    For the background, atmospheric events follow a softer spectrum,
    ie index ~ 3, while astrophysical a harder one, ie index ~ 2.
    You get the expected number of background events per energy bin
    from the respective spectra. The normalization of the spectra is wrt
    the total number of simulated events (ie for all energy bins).
    For the astrophysical background, the total number of events is defined
    such as the signal events amount to a given fraction of the
    astrophysical background events. This is to avoid having
    to take into account the background when signal is too high.
    The atmospheric background events are calculated by subtracting
    the number of simulated signal and astrophysical background events
    from the total number of events N that you specify.
    The positions of all background events are randomly drawn from a
    uniform distribution, with the DEC being constrained to the band
    defined when simulating the sources.

    The expected number of all signal events per source is calculated wrt
    a S**a power law, and you can select if all the sources
    have identical fluxes, or they are assigned randomly per source.
    Signal events also follow a hard spectrum, wrt which you estimate
    the expected number of signal events per energy bin per source,
    with the normalization of the spectrum done wrt the total number
    of expected events per source (ie you normalize per source).
    To simulate the signal events, we draw from a Poisson distribution
    around the expected number of events per energy bin.
    Their positions are simulated by first drawing from a Gaussian
    (shifted by the angular reconstruction error) for the events (RA, DEC)
    and a mock true (RA, DEC), which is then rotated to match the source position

    Args:
        src_pop (SimulateSources): the simulated source population
        N (int): number of total events in the dataset
        astro_fraction (float): fraction of signal to astrophysical background
                Used to define the number of astrophysical background events
                wrt to the number of signal events you get for given nsources.
                Default is 0.2, so all signal events amount to 20% of the astrophysical background
        identical (bool): if you want to simulate a population
                where all the sources have identical neutrino fluxes. Default is False
        smax (float): maximum neutrino flux in the population.
                Corresponds to the flux (here number of neutrinos)
                the brightest source in the population has. Default is 5
        scale (float, optional): the signal injection scale.
                This scales up or down the number of signal events in the dataset.
                Defaults to 1 for the "true" number of signal events
                (ie what you expect + Poisson noise)
        gamma (float, optional): spectral index for neutrino production model.
                Assume that all sources in the population produce neutrinos
                according to E**gamma power law, where E is the neutrino energy.
                Defaults to -1.9 for standard E^(-2) power law
        dnds_index (float, optional): index for the sources neutrino flux distribution.
                Defaults to -2.5 for an S^(-5/2) distribution.
        gamma_bkg (Tuple[float, float], optional): spectral indices for (atmospheric, astrophysical) background fluxes.
                Default is (-3, -1.9)
        min_e (float, optional): minimum energy in power of 10 and units of GeV.
                Default 3 (ie 10^3 GeV = 1 TeV)
        max_e (float, optional): maximum energy in power of 10 and units of GeV.
                Default is 6 (ie 10^6 GeV = 1 PeV)
        e_bins (int, optional): number of bins in [min_e, max_e] energy range.
                Default is 1 (ie full energy range)
        ang_error (float, optional): angular error in deg.
                Defaults to 0.8deg, which is the median value from the 10y NT MC (nt_v005_p01)
    """

    def __init__(
        self,
        src_pop: SimulateSources,
        N: int,
        smax: float = 5,
        identical: bool = False,
        scale: float = 1.0,
        astro_fraction: float = 0.2,
        gamma: float = -1.9,
        dnds_index: float = -2.5,
        gamma_bkg: Tuple[float, float] = (-3.0, -1.9),
        min_e: float = 3.0,
        max_e: float = 6.0,
        e_bins: int = 1,
        ang_error: float = 0.8,
        bkg_only: bool = False,
    ) -> None:

        self.srcs = src_pop
        self.nsrcs = src_pop.nsrcs
        self.n = N
        self.f_astro = astro_fraction
        self.min_dec = src_pop.min_dec
        self.max_dec = src_pop.max_dec
        self.angular_error = np.deg2rad(ang_error)
        self.Smax = smax
        self.identical_flux = identical
        self.scale = scale
        self.gamma = gamma
        self.flux_index = dnds_index
        self.gamma_atm = gamma_bkg[0]
        self.gamma_astro = gamma_bkg[1]
        self.min_e = min_e
        self.max_e = max_e
        self.nbins = e_bins
        self.scale=scale

        self.energy_bins = np.linspace(self.min_e, self.max_e, self.nbins + 1)

        # create the dataset
        self.calculate_sig()
        self.simulate_bkg_events()
        self.dataset = []
        for i in range(self.nbins):
            if not bkg_only:
                self.simulate_signal_events()
                tot_events_per_bin = self.bkg_pos[i] + self.sig_pos[i]
                self.dataset.append(tot_events_per_bin)
            else:
                tot_events_per_bin = self.bkg_pos[i]
                self.dataset.append(tot_events_per_bin)

    def rotate_position(
        self,
        ra1: float,
        dec1: float,
        ra2: float,
        dec2: float,
        ra3: float,
        dec3: float,
    ) -> Tuple[float, float]:
        """
        Rotate event position with reconstructed (ra1, dec1) and true (ra2, dec2)
        in a way that true position exactly maps onto source position (ra3, dec3),
        as if it was originally incident on the source (ie true position = source position).
        All angles are treated as radians.

        Args:
            ra1 (float): events RA
            dec1 (float): events DEC
            ra2 (float): events true RA
            dec2 (float): events true DEC
            ra3 (float): source Right Ascension
            dec3 (float): source Declination

        Returns:
            (float, float) new RA and Dec for the event
        """

        # Turns Right Ascension/Declination into Azimuth/Zenith for healpy
        phi1 = ra1 - np.pi
        zen1 = np.pi / 2.0 - dec1
        phi2 = ra2 - np.pi
        zen2 = np.pi / 2.0 - dec2
        phi3 = ra3 - np.pi
        zen3 = np.pi / 2.0 - dec3

        # Rotate ra1 and dec1 towards the pole?
        rot_matrix = hp.rotator.get_rotation_matrix((phi2, -zen2, 0.0))[0]
        x = hp.rotator.rotateDirection(rot_matrix, zen1, phi1)

        # Rotate towards ra3, dec3
        rot_pos = hp.rotator.rotateDirection(
            np.dot(
                hp.rotator.get_rotation_matrix((-phi3, 0, 0))[0],
                hp.rotator.get_rotation_matrix((0, zen3, 0.0))[0],
            ),
            x[0],
            x[1],
        )

        dec = np.pi / 2.0 - rot_pos[0]  # rot_pos[0] = zenith
        ra = rot_pos[1] + np.pi  # rot_pos[1] = azimuth

        return ra, dec

    def simulate_sig_events_pos(
        self, n_events: int, src_ra: float, src_dec: float
    ) -> List[Tuple[float, float]]:
        """
        Simulate positions for the signal events.
        First create the events RA & DEC (in rad) by randomly
        drawing from a Gaussian distribution with
        mean = 0 & std = 1, which we shift by the
        angular error (ie make angular error the distributions std).
        For the simulated RA we also add pi.
        The true events positions are also simulated
        to be pi and 0 for the RA, DEC respectively.
        Using these we rotate the events positions
        to match the source ones.
        Return a list of tuples containing the
        new (DEC, RA) for each event

        Args:
            n_events (int): number of signal events
            src_ra (float): the sources RA
            src_dec (float): the sources DEC

        Returns:
            [(float, float)]: list of n_events (DEC, RA)
        """

        sim_ra = np.pi + norm.rvs(size=n_events) * self.angular_error
        sim_dec = norm.rvs(size=n_events) * self.angular_error
        true_ra = np.ones_like(sim_dec) * np.pi
        true_dec = np.zeros_like(sim_dec)

        sig_events_pos = []
        for i in range(n_events):
            new_ra, new_dec = self.rotate_position(
                ra1=sim_ra[i],
                dec1=sim_dec[i],
                ra2=true_ra[i],
                dec2=true_dec[i],
                ra3=src_ra,
                dec3=src_dec,
            )
            sig_events_pos.append((new_dec, new_ra))

        return sig_events_pos

    def flux_distribution(
        self, nsources: Union[int, npt.NDArray]
    ) -> Union[int, npt.NDArray]:
        """The S**a flux distribution for the source population,
        in terms of number of neutrinos. It's parametrized
        as the number of neutrinos Smax for the brightest source
        in the population * number of sources


        Args:
            nsources (int | numpy array): number of sources
                Can be either a number, or an array of size nsources
                with initialized source fluxes

        Returns:
            source fluxes. Can be either a number, or a numpy array with fluxes for each source
        """

        return self.Smax * nsources ** (1.0 / (self.flux_index + 1.0))

    def calculate_sig(self) -> None:
        """
        Estimate the number of expected and injected
        signal neutrinos per energy bin according to
        the neutrino flux distribution.
        First initialize the fluxes,
        if all sources have identical fluxes or not,
        and then calculate them for each source
        following the distribution.
        For each source, the expected number
        of signal events per energy bin is estimated
        wrt the flux, while the injected
        (ie what is actually simulated) is drawn from a
        Poisson distribution around the expectation value.
        """

        # initialize fluxes
        if self.identical_flux:
            srcs = self.nsrcs * np.ones(self.nsrcs)
        else:
            srcs = self.nsrcs * np.random.rand(self.nsrcs)
        # calculate sources fluxes
        src_fluxes = self.flux_distribution(srcs)

        # calculate number of signal events per energy bin for each source
        self.exp_sig_per_bin = np.zeros((self.nsrcs, len(self.energy_bins) - 1))
        self.inj_sig_per_bin = np.zeros((self.nsrcs, len(self.energy_bins) - 1))
        # normalize fluxes by dividing with the full energy range
        g1 = self.gamma + 1
        norm_fluxes = (
            g1
            * src_fluxes
            / (10 ** (self.energy_bins[-1] * g1) - 10 ** (self.energy_bins[0] * g1))
        )

        for i in range(self.nbins):
            # for all sources calculate the flux within the i-th energy bin
            self.exp_sig_per_bin[:, i] = (
                norm_fluxes
                / g1
                * (
                    10 ** (self.energy_bins[i + 1] * g1)
                    - 10 ** (self.energy_bins[i] * g1)
                )
            )
            self.inj_sig_per_bin[:, i] = np.random.poisson(
                self.exp_sig_per_bin[:, i] * self.scale
            )

        # self.all_sig_events_per_src = self.sig_events_per_bin.sum(1)
        self.all_sig_events_per_bin = self.inj_sig_per_bin.sum(0)
        self.tot_exp_sig_per_bin = self.exp_sig_per_bin.sum(0)

    def simulate_signal_events(self) -> None:
        """
        Simulate the signal events in the dataset.
        For each src, the positions of n signal events
        are simulated, where n is the number
        of injected signal events per bin.
        The (DEC, RA) for these events are first drawn
        from a Gaussian distribution and then
        rotated to match the source's position.
        """

        # simulate signal events positions as in (DEC, RA)
        self.sig_pos = []
        for i in range(self.nbins):
            sig_pos_per_bin = []
            for n in range(self.nsrcs):
                # for each src get the number of signal events per bin
                sig_per_src_per_bin = self.inj_sig_per_bin[n][i]
                # get the src position
                src_dec, src_ra = (
                    self.srcs.src_positions[n][0],
                    self.srcs.src_positions[n][1],
                )
                # simulate (DEC, RA) for all signal events per src per bin
                sig_pos_per_src_per_bin = self.simulate_sig_events_pos(
                    int(sig_per_src_per_bin), src_ra, src_dec
                )

                sig_pos_per_bin.extend(sig_pos_per_src_per_bin)
            self.sig_pos.append(sig_pos_per_bin)

    def simulate_bkg_events(self) -> None:
        """
        Simulate atmospheric and astrophysical background events.
        The number of astrophysical background events n_astro is estimated
        wrt the total number of signal events and the chosen fraction.
        The flux normalization is done by using n_astro & n_atm
        respectively (ie total number of events in the full energy range),
        the number of events per energy bin are calculated
        according to their respective spectral distributions, and
        by adding Poisson noise we end up with the
        expected number of background events per energy bin.
        We simulate their positions by drawing the DEC & RA
        from uniform distributions, where the DEC must be
        within the source population DEC bandwidth.
        """

        # total number of astrophysical background events wrt the
        # total injected number of signal events (ie for all sources & all energy bins)
        all_sig_events = sum(self.all_sig_events_per_bin)
        self.n_astro = int(all_sig_events / self.f_astro)
        self.n_atm = self.n - self.n_astro - all_sig_events

        # calculate number of bkg events per energy bin
        self.atm_events_per_bin = np.zeros(len(self.energy_bins) - 1)
        self.astro_events_per_bin = np.zeros(len(self.energy_bins) - 1)
        # normalization factors
        g1_atm = self.gamma_atm + 1
        g1_astro = self.gamma_astro + 1
        norm_atm = (
            g1_atm
            * self.n_atm
            / (
                10 ** (self.energy_bins[-1] * g1_atm)
                - 10 ** (self.energy_bins[0] * g1_atm)
            )
        )
        norm_astro = (
            g1_astro
            * self.n_astro
            / (
                10 ** (self.energy_bins[-1] * g1_astro)
                - 10 ** (self.energy_bins[0] * g1_astro)
            )
        )

        for i in range(self.nbins):
            # calculate number of events per energy bin
            atm_flux_per_bin = (
                norm_atm
                / g1_atm
                * (
                    10 ** (self.energy_bins[i + 1] * g1_atm)
                    - 10 ** (self.energy_bins[i] * g1_atm)
                )
            )
            self.atm_events_per_bin[i] = np.random.poisson(atm_flux_per_bin)
            astro_flux_per_bin = (
                norm_astro
                / g1_astro
                * (
                    10 ** (self.energy_bins[i + 1] * g1_astro)
                    - 10 ** (self.energy_bins[i] * g1_astro)
                )
            )
            self.astro_events_per_bin[i] = np.random.poisson(astro_flux_per_bin)

        # simulate RA & DEC for all background events per energy bin
        self.bkg_pos = []
        for i in range(self.nbins):
            tot_bkg_events_per_bin = (
                self.atm_events_per_bin[i] + self.astro_events_per_bin[i]
            )
            bkg_pos_per_bin = [
                (
                    np.deg2rad(np.random.uniform(self.min_dec, self.max_dec)),
                    np.deg2rad(np.random.uniform(0, 360)),
                )
                for _ in range(int(tot_bkg_events_per_bin))
            ]
            self.bkg_pos.append(bkg_pos_per_bin)


class LLH:
    """
    Construct the llh to minimize.

    The signal and background pdfs have only the spatial terms,
    where the signal spatial pdf is a Gaussian that depends only on the opening angle
    (ie angular distance between simulated events and source positions)
    since the angular error is fixed, while the background pdf
    is just a uniform distribution of the events positions.
    In this implementation, we use the full dataset
    to calculate the opening angles for each source.

    The TS formula as the ratio of signal to background pdfs
    is constructed per energy bin and minimized wrt the signal events n_s.
    In this implementation, the minimization takes place
    in each bin where the best-fit n_s is estimated, so that
    the total n_s for the full energy range is simply
    the sum of the best-fit values for each energy bin.

    Args:
        data (SimulateDataset): instance of the simulated dataset
        srcs (SimulateSources): instance of the simulated source population
        sig_spatial_threshold (float, optional): threshold value for the
            signal spatial pdf. Used to get rid of very small values
            that will not contribute to the llh. Defaults to 1e-21
            (an arbitrarily small value)
    """

    def __init__(
        self,
        data: SimulateDataset,
        srcs: SimulateSources,
        sig_spatial_threshold: float = 1e-21,
    ) -> None:

        self.e_bins = data.energy_bins
        self.ang_err = data.angular_error
        self.sig_spatial_threshold = sig_spatial_threshold

        self.data_pos = data.dataset
        self.src_pos = srcs.src_positions

    @staticmethod
    def opening_angles(
        src_dec: float, src_ra: float, data_dec: float, data_ra: float
    ) -> float:
        """Function for calculating the angular distance
        between the source & event positions, both in DEC, RA

        Args:
            src_dec (float): source declination (in rad)
            src_ra (float): source right ascention (in rad)
            data_dec (float): event declination (in rad)
            data_ra (float): event right ascention (in rad)

        Returns:
            float: the opening angle (in rad)
        """

        oa = np.arccos(
            np.sin(src_dec) * np.sin(data_dec)
            + np.cos(src_dec) * np.cos(data_dec) * np.cos(src_ra - data_ra)
        )
        return oa

    def compute_opening_angles_per_src(
        self,
        src_pos: Tuple[float, float],
        data_pos: List[Tuple[float, float]],
    ) -> npt.NDArray:
        """Compute the opening angles between the source and all the events positions

        Args:
            src_pos (Tuple[float, float]): source position in (DEC, RA)
            data_pos (List[Tuple[float, float]]): list of events positions in (DEC, RA)

        Returns:
            npt.NDArray: array of size = len(data) with opening angles
        """

        src_dec = src_pos[0]
        src_ra = src_pos[1]
        op_angles = np.zeros(len(data_pos))
        for i, pos in enumerate(data_pos):
            data_dec = pos[0]
            data_ra = pos[1]
            op_angles[i] = self.opening_angles(src_dec, src_ra, data_dec, data_ra)
        return op_angles

    def spatial_pdf(self, opening_angles: npt.NDArray) -> npt.NDArray:
        """Calculate the signal spatial pdf.
        The pdf is chosen as a 2D Gaussian that depends on the
        angular distances and the angular error

        Args:
            opening_angles (npt.NDArray): opening angles per source

        Returns:
            npt.NDArray: spatial pdf values per source
        """
        return (1 / (2 * np.pi * self.ang_err**2)) * np.exp(
            -(opening_angles**2) / (2 * self.ang_err**2)
        )

    def calculate_ts_per_bin(
        self, ns: float, weights: npt.ArrayLike, data_per_bin: npt.ArrayLike
    ) -> float:
        """Construct the TS function per energy bin.

        First calculate the opening angles per source wrt
        the data in a given bin to calculate the
        signal spatial pdfs per source. We use only the
        ones that are above a threshold value, since
        for values below the event won't contribute to the llh.

        Construct the function with the ns as free parameter.
        For each source, the ns is weighted wrt
        the expected number of signal events in that energy bin,
        so that we actually minimize wrt the number of signal events.
        Since we get the log of the llh for all the events, as per
        the definition, we set the llh to 0.99 when is negative or 0
        to avoid errors. A llh is negative when the ns * weight/n > 1,
        where n = total number of events in the bin, but the signal
        spatial pdf is low, so we basically assume that
        the event is background by setting llh = 0.99.
        We sum the log of llhs over all events to get the llh per source,
        and then sum again to get the TS value for all sources

        Args:
            n_s (float): injected number of neutrinos
            weights (npt.ArrayLike): expected number of signal events
                for all the sources in given bin
            data_per_bin (npt.ArrayLike): data for given energy bin
                containing events positions as in (DEC, RA)

        Returns:
            float: the TS value for a given ns in a given energy bin
        """

        llh_vals = []
        for i, src in enumerate(self.src_pos):
            # compute array per src, each element of the array
            # is the opening angle wrt each event in data
            opening_angles = self.compute_opening_angles_per_src(src, data_per_bin)
            # compute signal spatial pdf for each src using data
            sig_spatial = self.spatial_pdf(opening_angles)
            sig_spatial_mask = sig_spatial > self.sig_spatial_threshold
            sig_spatial = sig_spatial[sig_spatial_mask]

            # calculate llh value for each src and all the data
            # bkg spatial pdf = 1/4*pi
            llh = 1 + (ns * weights[i] / len(data_per_bin)) * (
                4 * np.pi * sig_spatial - 1.0
            )
            llh_value = np.where(llh <= 0, 0.99, llh)
            llh_per_src = np.sum(np.log(llh_value))
            llh_vals.append(llh_per_src)

        return 2.0 * np.sum(llh_vals)


class Analysis:
    """Perform the analysis once, to get the best-fit
    number of signal events ns and the corresponding TS value.
    The minimization is done per energy bin, where you fit
    the ns to the data in that energy bin, and subsequently
    calculate the TS for that best-fit value.
    For the full energy range, you just add the best-fit values.
    If the best-fit TS is negative, set it to 0.

    Args:
        llh (LLH): instance of the llh function
        init_ns (float, optional): initial guess for the injected ns.
            Passed in the minimizer as an array. Defaults to 1.0.
        ns_bounds (Tuple[float, float], optional): bounds for the ns
            Passed in the minimizer as an array. Defaults to (0, 1000).
        tot_weights (npt.ArrayLike): contains the expected number of
            signal events for all sources per energy bin.
    """

    def __init__(
        self,
        llh: LLH,
        tot_weights: npt.ArrayLike,
        init_ns: float = 1.0,
        ns_bounds: Tuple[float, float] = (0, 1000),
    ) -> None:

        self.llh = llh
        self.weights = tot_weights
        self.start_seed = init_ns
        self.bounds = ns_bounds

        self.dataset = llh.data_pos
        self.srcs = llh.src_pos

    def minimize_llh_func(
        self, data: npt.ArrayLike, w: npt.ArrayLike
    ) -> Tuple[Optional[npt.NDArray], Optional[float]]:
        """Minimize the llh function wrt ns for given energy bin.
        First make the function to be minimized with ns as the parameter,
        ie construct the TS for given data and weights (per bin).
        Then this function is minimized, where ns takes different values
        within the given bounds for each step of the minimization,
        starting with the initial guess.

        Args:
            data (npt.ArrayLike): the events positions in the energy bin
            w (npt.ArrayLike): weights array passed in the llh.
                Contains the expected number of signal events
                for all the sources in the energy bin

        Returns:
            (best_fit_ns, best_ts). If minimization is successful,
                minimizer returns best-fit value as numpy array, and
                for that best-fit ns the TS is calculated.
                If not successful, both are None
        """

        def llh_func(ns: npt.NDArray) -> Callable:
            """Function to minimize.
            The ns parameter should be a numpy array
            to conform with the scipy minimize requirements.
            Returns TS function

            Args:
                ns (npt.NDArray): number of signal events

            Returns:
                TS function for given data and ns as free parameter
            """
            n_s = ns[0]
            return self.llh.calculate_ts_per_bin(n_s, w, data)

        result = minimize(llh_func, [self.start_seed], bounds=[self.bounds])
        if result.success:  # if minimization is successful
            best_ns = result.x  # array with best-fit ns
            best_ts = llh_func(best_ns)
            if best_ts < 0 or best_ts == -0.0:
                best_ts = 0.0
        else:
            print(f"Minimizer failed with {result.message}")
            best_ns, best_ts = None, None

        return best_ns, best_ts

    def find_total_ns(self) -> Tuple[float, float]:
        """Get best-fit ns and TS values
        for the full energy range

        Returns:
            Tuple[float, float]: (total ns, total TS)
        """
        ts = 0
        ns = 0
        for i, data in enumerate(self.dataset):
            if len(data) == 0:
                continue
            else:
                w = self.weights[i]
                ns_per_bin, ts_per_bin = self.minimize_llh_func(data, w)
            if ns_per_bin is not None and ts_per_bin is not None:
                ns += np.sum(ns_per_bin)
                ts += ts_per_bin
            else:
                raise ValueError("Minimization went wrong, abort mission")

        return ns, ts


if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-nsrcs",
        type=int,
        required=True,
        help="Number of sources to simulate",
    )
    parser.add_argument(
        "-N",
        type=int,
        required=True,
        help="Total number of events in the dataset",
    )
    parser.add_argument(
        "-dec_band",
        type=float,
        help="The declination band (in deg) for which the source positions are simulated. Default is +-60deg",
        default=60,
    )
    parser.add_argument(
        "-smax",
        type=float,
        help="The maximum number of neutrinos for the brightest source in the population. Default is 5",
        default=5.0,
    )
    parser.add_argument(
        "-identical_fluxes",
        type=ast.literal_eval,
        help="Whether or not the sources have identical neutrino fluxes. Default is False",
        default=False,
    )
    parser.add_argument(
        "-scale", type=float, default=1.0, help="Scale for signal injection"
    )
    parser.add_argument(
        "-flux_index",
        type=float,
        help="The index for the sources' flux distribution. Default is -2.5, corresponding to S^(-5/2) distribution",
        default=-2.5,
    )
    parser.add_argument(
        "-gamma",
        type=float,
        help="The spectral index for the power-law neutrino spectrum. Default is -1.9, corresponding to the standard E^(-2) production",
        default=-1.9,
    )
    parser.add_argument(
        "-fastro",
        type=float,
        help="The fraction of signal to astrophysical background events. Default is 20%",
        default=0.2,
    )
    parser.add_argument(
    "-gamma_bkg",
    nargs=2,
    type=float,
    help="The spectral indices for the atmospheric and astrophysical background fluxes respectively. Default is (-3., -1.9)",
    default=[-3.0, -1.9],
)

    parser.add_argument(
        "-min_e",
        type=float,
        help="Minimum energy as in power of 10 (in GeV). Default is 3, corresponding to 10^3 GeV = 1 TeV",
        default=3.0,
    )
    parser.add_argument(
        "-max_e",
        type=float,
        help="Maximum energy as in power of 10 (in GeV). Default is 6, corresponding to 10^6 GeV = 1 PeV",
        default=6.0,
    )
    parser.add_argument(
        "-bins",
        type=int,
        help="Number of energy bins. Default is 1, corresponding to the full energy range [10**min_e, 10**max_e]",
        default=1,
    )
    parser.add_argument(
        "-sigma",
        type=float,
        help="The angular error (in deg). Default is 0.8, corresponding to the median value from the 10y NT MC",
        default=0.8,
    )
    parser.add_argument(
        "-sig_spat_thresh",
        type=float,
        help="Threshold for the signal spatial pdf, above which the event is counted as signal. Default is 10^(-21)",
        default=1e-21,
    )
    parser.add_argument(
        "-ns_seed",
        type=float,
        help="Initial guess for the ns passed in the minimizer. Default is 1",
        default=1.0,
    )
    parser.add_argument(
    "-bounds",
    nargs=2,
    type=float,
    help="Bounds for the ns passed in the minimizer. Default is (0, 1000)",
    default=[0, 1000],
)

    parser.add_argument(
        "-ntrials",
        type=int,
        help="Number of trials to perform. Default is 1",
        default=1,
    )
    parser.add_argument(
    "-scalefactormin",
    type=float,
    default=1,
    help="Minimum scale factor for signal injection",
    )
    parser.add_argument(
        "-scalefactormax",
        type=float,
        default=10.0,
        help="Maximum scale factor for signal injection",
    )
    parser.add_argument(
        "-nscale",
        type=int,
        default=10,
        help="Number of scale steps between min and max",
    )
    parser.add_argument(
        "-only_bkg",
        type=ast.literal_eval,
        help="Do you want to run background-only trials?",
        default=False,
    )
    parser.add_argument(
        "-load_srcs",
        type=ast.literal_eval,
        help="Do you want to load the source population?",
        default=False,
    )
    parser.add_argument(
        "-save_srcs",
        type=ast.literal_eval,
        help="Do you want to save the source population?",
        default=False,
    )
    parser.add_argument(
        "-save_path",
        type=str,
        default=".",
        help="Path where you want to save the results",
    )
    parser.add_argument(
        "-sfx",
        type=str,
        help="Suffix for the results file name",
    )
    parser.add_argument(
        "-rm_res",
        type=ast.literal_eval,
        default=False,
        help="Do you want to remove old results?",
    )

    args = parser.parse_args()

    if args.save_path == ".":
        savepath = os.getcwd()
    else:
        savepath = args.save_path
    if args.sfx is not None:
        sfx = "_" + args.sfx
    else:
        sfx = ""

    src_path = os.path.join(savepath, "src_pop" + sfx + ".pkl")
    res_path = os.path.join(savepath, "results" + sfx + ".pkl")

    if os.path.isfile(res_path) and not args.rm_res and not args.load_srcs:
        raise ValueError(
            "When adding trials you need to load the same source population"
        )

    # simulate the srcs outside the loop so their positions don't change
    srcs = SimulateSources(
        nsources=args.nsrcs, min_dec=-args.dec_band, max_dec=args.dec_band
    )
    if args.save_srcs:
        with open(src_path, "wb") as sp:
            pickle.dump(srcs, sp)
    if args.load_srcs:
        assert os.path.isfile(src_path)
        with open(src_path, "rb") as sf:
            srcs = pickle.load(sf)

            
    # load files if there are already some
    if os.path.isfile(res_path) and not args.rm_res:
        with open(res_path, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    if args.only_bkg:
        scales = [1.0]  # only use one scale for the background
    else:
        scales = np.linspace(args.scalefactormin, args.scalefactormax, args.nscale)

    # simulation and analysis for all the scales
    for scale in scales:
        all_ns, all_TS, n_inj, n_exp = [], [], [], []

        for i in range(args.ntrials):
            # create new dataset for each trial, except the sources
            data = SimulateDataset(
                src_pop=srcs,
                smax=args.smax,
                identical=args.identical_fluxes,
                scale=scale,
                N=args.N,
                astro_fraction=args.fastro,
                gamma=args.gamma,
                dnds_index=args.flux_index,
                gamma_bkg=tuple(args.gamma_bkg),
                min_e=args.min_e,
                max_e=args.max_e,
                e_bins=args.bins,
                ang_error=args.sigma,
                bkg_only=args.only_bkg,
            )

            # save data
            n_inj.append(sum(data.all_sig_events_per_bin))
            n_exp.append(sum(data.tot_exp_sig_per_bin))

            # analyse the dataset
            llh = LLH(data, srcs, args.sig_spat_thresh)
            weights = [data.exp_sig_per_bin[:, i] for i in range(args.bins)]
            ana = Analysis(llh, weights, args.ns_seed, args.bounds)
            ns, ts = ana.find_total_ns()
            all_ns.append(ns)
            all_TS.append(ts)

        # save of the results for all scales
        if scale not in results:
            results[scale] = {"n_inj": [], "n_exp": [], "ns": [], "TS": []}

        results[scale]["n_inj"].extend(n_inj)
        results[scale]["n_exp"].extend(n_exp)
        results[scale]["ns"].extend(all_ns)
        results[scale]["TS"].extend(all_TS)

    # save results in a pkl file
    with open(res_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Ergebnisse erfolgreich in {res_path} gespeichert.")
