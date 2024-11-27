#!/usr/bin/env python

""""
This is an example script for running the full analysis for 1 source and a single energy bin.
You need to simulate the dataset, ie the number of total events, and the source(s) position(s)
in order to formulate the llh. When you create the llh and subsequently the TS, you need to
minimize the TS wrt the number of signal events n_s, in order to find the best-fit ns.
You do this many times to find the TS distribution,
each time with a different dataset (background RA are randomly selected each time)

For this energy bin, you first simulate the background events n_bkg,
for 2 different spectral indices and a uniform RA & DEC. For the same energy range,
you also simulate the signal events n_sig: for a source at a given position that is drawn from
uniform RA & DEC distributions, its neutrino flux follows an .
If you have many sources, then their E^(-gamma) neutrino fluxes , which 
identical (ie same flux for all), or varying emission (ie randomly assign flux).
For both the background & signal, you convert from fluxes to number of events
by adding Poisson noise, so then you end up with a dataset of N = n_bkg + n_sig events.
Now for each event in the dataset you assign RA & DEC drawn from Gaussian distributions,
and you select the spatially coincident events for each source which you use to
calculate the pdfs in the llh. You then minimize the llh with respect to the signal

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
    The simulated dataset consists of the events positions in (RA, DEC)
    for each energy bin.

    For the background, atmospheric events follow a softer spectrum,
    ie index ~ 3, while astrophysical a harder one, ie index ~ 2.
    You get the expected number of background events per energy bin
    from the respective spectra. The normalization of the spectra is wrt
    the total number of events (ie for all energy bins) you want to simulate.
    For the atmospheric background, this is the n_atmo parameter.
    For the astrophysical background, the total number of events is defined
    such as the signal events amount to a given fraction of the
    astrophysical background events. This is to avoid having
    to take into account the background when signal is too high.
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
        n_atmo (int): number of background atmospheric events
        astro_fraction (float): fraction of signal to astrophysical background
                Used to define the number of astrophysical background events
                wrt to the number of signal events you get for given nsources.
                Default is 0.2, so all signal events amount to 20% of the astrophysical background
        identical (bool): if you want to simulate a population
                where all the sources have identical neutrino fluxes. Default is False
        smax (float): maximum neutrino flux in the population.
                Corresponds to the flux (here number of neutrinos)
                the brightest source in the population has. Default is 5
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
        n_atm: int,
        smax: float = 5,
        identical: bool = False,
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
        self.n_atm = n_atm
        self.astro_fraction = astro_fraction
        self.min_dec = src_pop.min_dec
        self.max_dec = src_pop.max_dec
        self.angular_error = ang_error
        self.Smax = smax
        self.identical_flux = identical
        self.gamma = gamma
        self.flux_index = dnds_index
        self.gamma_atm = gamma_bkg[0]
        self.gamma_astro = gamma_bkg[1]
        self.min_e = min_e
        self.max_e = max_e
        self.nbins = e_bins

        self.energy_bins = np.linspace(self.min_e, self.max_e, self.nbins + 1)

        # create the dataset
        self.simulate_signal_events()
        self.simulate_bkg_events()
        self.dataset = []
        for i in range(self.nbins):
            if not bkg_only:
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

    def simulate_signal_events(self) -> None:
        """
        Simulate the signal events according to the neutrino flux per energy bin.
        First initialize the fluxes, if all sources have identical fluxes or not,
        and then calculate them for each source following the distribution.
        For each source, the expected number of signal events per energy bin are estimated
        wrt the flux, and then drawn from a Poisson distribution.
        The DEC & RA for these events are randomly drawn from a uniform distribution,
        where the DEC must be within the source population DEC bandwidth.
        """

        # initialize fluxes
        if self.identical_flux:
            srcs = self.nsrcs * np.ones(self.nsrcs)
        else:
            srcs = self.nsrcs * np.random.rand(self.nsrcs)
        # calculate source fluxes
        src_fluxes = self.flux_distribution(srcs)

        # calculate number of signal events per energy bin for each source
        sig_flux_per_bin = np.zeros((self.nsrcs, len(self.energy_bins) - 1))
        self.sig_events_per_bin = np.zeros((self.nsrcs, len(self.energy_bins) - 1))
        # normalize fluxes by dividing with the full energy range
        g1 = self.gamma + 1
        norm_fluxes = (
            g1
            * src_fluxes
            / (10 ** (self.energy_bins[-1] * g1) - 10 ** (self.energy_bins[0] * g1))
        )

        for i in range(self.nbins):
            # for all sources calculate the flux within the i-th energy bin
            sig_flux_per_bin[:, i] = (
                norm_fluxes
                / g1
                * (
                    10 ** (self.energy_bins[i + 1] * g1)
                    - 10 ** (self.energy_bins[i] * g1)
                )
            )
            self.sig_events_per_bin[:, i] = np.random.poisson(sig_flux_per_bin[:, i])

        # self.all_sig_events_per_src = self.sig_events_per_bin.sum(1)

        # simulate signal events positions as in (DEC, RA)
        self.all_sig_events_per_bin = self.sig_events_per_bin.sum(0)
        self.sig_pos = []
        for i in range(self.nbins):
            sig_pos_per_bin = []
            for n in range(self.nsrcs):
                # for each src get the number of signal events per bin
                sig_per_src_per_bin = self.sig_events_per_bin[n][i]
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

        # total number of astrophysical background events wrt to
        # the total number of signal events (ie for all sources & all energy bins)
        self.n_astro = int(sum(self.all_sig_events_per_bin) / self.astro_fraction)

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
    where the signal spatial pdf is a 2D Gaussian that depends on the opening angle
    (ie angular distance between simulated events and source positions)
    and the angular error that is fixed, while the background pdf
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
        ang_error: float = 0.8,
        sig_spatial_threshold: float = 1e-21,
    ) -> None:

        self.e_bins = data.energy_bins
        self.ang_err = np.deg2rad(ang_error)
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
        """Compute the opening angles between the source and all the data positions

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

    def calculate_ts_per_bin(self, n_s: float, data_per_bin: npt.ArrayLike) -> float:
        """Construct the TS function per energy bin.

        First calculate the opening angles per source wrt
        the data in a given bin to calculate the
        signal spatial pdfs per source. We use only the
        ones that are above a threshold value, since
        for values below the event won't contribute to the llh.

        Construct the function with the n_s as free parameter,
        this is the number of signal neutrinos you inject each time and
        let it vary until you find the best-fit value.
        You sum the log of the expression to get the sum over all events,
        basically get the llh for each source, and then sum again
        to get the TS value for all sources

        Args:
            n_s (float): injected number of neutrinos
            data (npt.ArrayLike): data for given energy bin
                containing events positions as in (DEC, RA)

        Returns:
            float: the TS value for a given ns in a given energy bin
        """

        N = len(data_per_bin)
        llh_vals = []
        for src in self.src_pos:
            # compute array per src, each element of the array
            # is the opening angle wrt each event in data
            opening_angles = self.compute_opening_angles_per_src(src, data_per_bin)
            # compute signal spatial pdf for each src using data
            sig_spatial = self.spatial_pdf(opening_angles)
            # print("sig spatial before mask ", sig_spatial)
            sig_spatial_mask = sig_spatial > self.sig_spatial_threshold
            sig_spatial = sig_spatial[sig_spatial_mask]
            # print("sig spatial after mask ", sig_spatial)

            # calculate llh value for each src and all the data
            # bkg spatial pdf = 1/4*pi
            llh_value = 1 + (n_s / N) * (4 * np.pi * sig_spatial - 1.0)
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
    If the best-fit TS is negative, set it to 0 but flag it first.

    Args:
        llh (LLH): instance of the llh function
        init_ns (float, optional): initial guess for the ns.
            Passed in the minimizer as an array. Defaults to 1.0.
        ns_bounds (Tuple[float, float], optional): bounds for the ns
            Passed in the minimizer as an array. Defaults to (0, 1000).
    """

    def __init__(
        self,
        llh: LLH,
        init_ns: float = 1.0,
        ns_bounds: Tuple[float, float] = (0, 1e3),
    ) -> None:

        self.llh = llh
        self.start_seed = init_ns
        self.bounds = ns_bounds

        self.dataset = llh.data_pos
        self.srcs = llh.src_pos

    def minimize_llh_func(
        self, data: npt.ArrayLike
    ) -> Tuple[Optional[npt.NDArray], Optional[float]]:
        """Minimize the llh function wrt ns.
        First make the function by constructing
        the TS as a function of ns for given data (per bin),
        and then minimize it with the provided
        initial guess and bounds for the ns.

        Args:
            data (npt.ArrayLike): the events positions in an energy bin

        Returns:
            (best_fit_ns, best_ts). If minimization is successful,
                minimizer returns best-fit value as numpy array, and
                for that best-fit ns the TS is calculated.
                If not successful, both are None
        """

        def llh_func(ns: npt.NDArray) -> Callable:
            """Function to minimize
            The ns parameter should be a numpy array
            to conform with the scipy minimize requirements.
            Returns (- TS function) because you want to
            minimize it (instead of maximizing the TS function)

            Args:
                ns (npt.NDArray): number of signal events

            Returns:
                -TS function for given data and ns as free parameter
            """
            n_s = ns[0]
            return self.llh.calculate_ts_per_bin(n_s, data)

        result = minimize(llh_func, [self.start_seed], bounds=[self.bounds])
        if result.success:  # if minimization is successful
            best_fit_ns = result.x  # array with best-fit ns
            best_ts = llh_func(best_fit_ns)
            if best_ts < 0 or best_ts == -0.0:
                best_ts = 0.0
        else:
            best_fit_ns, best_ts = None, None

        return best_fit_ns, best_ts

    def find_total_ns(self) -> Tuple[float, float]:
        """Get best-fit ns and TS values
        for the full energy range

        Returns:
            Tuple[float, float]: (total ns, total TS)
        """
        ts = 0
        ns = 0
        for data in self.dataset:
            ns_per_bin, ts_per_bin = self.minimize_llh_func(data)
            if ns_per_bin is not None and ts_per_bin is not None:
                ns += np.sum(ns_per_bin)
                ts += ts_per_bin
            else:
                raise ValueError("Minimization went wrong, abort mission")

        return ns, ts


if __name__ == "__main__":
    # configure how you run the script
    parser = argparse.ArgumentParser()

    # make the arguments with which to run the script
    parser.add_argument(
        "-nsrcs",
        type=int,
        required=True,
        help="Number of sources to simulate",
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
        "-natm",
        type=int,
        required=True,
        help="Number of atmospheric background events to simulate",
    )
    parser.add_argument(
        "-astro_fraction",
        type=float,
        help="The fraction of signal to astrophysical background events. Default is 20%",
        default=0.2,
    )
    parser.add_argument(
        "-gamma_bkg",
        type=Tuple[float, float],
        help="The spectral indices for the atmospheric and astrophysical background fluxes respectively. Default is (-3., -1.9)",
        default=(-3.0, -1.9),
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
        "-ang_err",
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
        "-ns_bounds",
        type=Tuple[float, float],
        help="Bounds for the ns passed in the minimizer. Default is (0, 1000)",
        default=(0.0, 1e3),
    )
    parser.add_argument(
        "-ntrials",
        type=int,
        help="Number of trials to perform. Default is 1",
        default=1,
    )
    parser.add_argument(
        "-save_path",
        type=str,
        required=True,
        help="Path where you want to save the results",
    )

    args = parser.parse_args()

    # simulate the srcs outside the loop so their positions don't change
    srcs = SimulateSources(
        nsources=args.nsrcs, min_dec=-args.dec_band, max_dec=args.dec_band
    )
    all_ns, all_TS = [], []
    for _ in range(args.ntrials):
        # for each trial make new dataset
        data = SimulateDataset(
            src_pop=srcs,
            smax=args.smax,
            identical=args.identical_fluxes,
            n_atm=args.natm,
            astro_fraction=args.astro_fraction,
            gamma=args.gamma,
            dnds_index=args.flux_index,
            gamma_bkg=args.gamma_bkg,
            min_e=args.min_e,
            max_e=args.max_e,
            e_bins=args.bins,
        )

        # perform analysis for this dataset
        llh = LLH(data, srcs)
        ana = Analysis(llh)
        ns, ts = ana.find_total_ns()
        all_ns.append(ns)
        all_TS.append(ts)

    # save results in a pickle file at the provided path
    filepath = os.path.join(args.save_path, "results.pkl")
    with open(filepath, "wb") as f:
        pickle.dump([all_ns, all_TS], f)
