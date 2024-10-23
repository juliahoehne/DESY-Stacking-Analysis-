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

import numpy as np
import numpy.typing as npt

from typing import Tuple, Optional, List, Mapping, Union

import matplotlib.pyplot as plt

# from scipy.optimize import curve_fit
import pandas as pd


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
    The positions of all the events are randomly distributed,
    but their energies follow different power laws.

    For the background, atmospheric events follow a softer spectrum,
    ie index ~ 3, while astrophysical a harder one, ie index ~ 2.
    You get the number of background events per energy bin
    from the respective fluxes according to the 2 different spectral assumptions.
    For the astrophysical background, the number of events is defined such as
    the signal events amount to a given fraction of the
    astrophysical background events. This is to avoid having to take into account
    the background when signal events is too high.

    Signal events have energies that follow a hard spectrum as well,
    but their positions are constrained in a box around the source positions.
    The number of signal events is defined by the flux of the sources:
    their neutrino fluxes are distributed according to a S**a power law,
    and you can select if all the sources have identical fluxes, or
    they are assigned randomly per source.

    For both background and signal events you add Poisson noise
    in the number of events you got from the fluxes, so by adding them you get the
    total number of events in the dataset.


    Args:
        src_pop (SimulateSources): the simulated source population
        n_atmo (int): number of background atmospheric events
        astro_fraction (float): fraction of signal to astrophysical background
                Used to define the number of astrophysical background events
                wrt to the number of signal events you get for given nsources.
                Default is 0.2, so all signal events amount to 20% of the astrophysical background
        identical (bool): if you want to simulate a population where all the sources
                have identical neutrino fluxes
        smax (float): maximum neutrino flux in the population.
                Corresponds to the flux (here number of neutrinos) the brightest
                source in the population has
        gamma (float, optional): The spectral index for the neutrino production model.
                Assume that all sources in the population produce neutrinos according to
                E**gamma power law, where E is the neutrino energy.
                Defaults to -1.9 for standard E^(-2) power law
        dnds_index (float, optional): The index for the neutrino flux distribution.
                Defaults to -2.5 for an S^(-5/2) distribution.
        gamma_bkg (Tuple[float, float], optional): spectral indices for (atmospheric, astrophysical) background fluxes,
                Default is (-3, -1.9)
        min_e (float, optional): minimum energy in power of 10 and units of GeV. Default 3 (ie 10^3 GeV = 1 TeV)
        max_e (float, optional): maximum energy in power of 10 and units of GeV. Default is 6 (ie 10^6 GeV = 1 PeV)
        e_bins (int, optional): number of bins in [min_e, max_e] energy range. Default is 1 (ie full energy range)
    """

    def __init__(
        self,
        src_pop: SimulateSources,
        smax: float,
        identical: bool,
        n_atm: int,
        astro_fraction: float = 0.2,
        gamma: float = -1.9,
        dnds_index: float = -2.5,
        gamma_bkg: Tuple[float, float] = (-3.0, -1.9),
        min_e: float = 3.0,
        max_e: float = 6.0,
        e_bins: int = 1,
    ) -> None:

        self.nsrcs = src_pop.nsrcs
        self.n_atm = n_atm
        self.astro_fraction = astro_fraction
        self.min_dec = src_pop.min_dec
        self.max_dec = src_pop.max_dec
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
            tot_events_per_bin = self.bkg_pos[i] + self.sig_pos[i]
            self.dataset.append(tot_events_per_bin)

    def flux_distribution(
        self, nsources: Union[int, npt.NDArray]
    ) -> Union[int, npt.NDArray]:
        """The S**a flux distribution for the source population,
        parametrized in terms of number of neutrinos * number of sources
        and normalized to the number of neutrinos Smax
        for the brightest source in the population

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

        # simulate DEC & RA for all signal events per energy bin
        self.all_sig_events_per_bin = self.sig_events_per_bin.sum(0)
        self.sig_pos = []
        for i in range(self.nbins):
            # simulate positions as (DEC, RA)
            sig_pos_per_bin = [
                (
                    np.deg2rad(np.random.uniform(self.min_dec, self.max_dec)),
                    np.deg2rad(np.random.uniform(0, 360)),
                )
                for _ in range(int(self.all_sig_events_per_bin[i]))
            ]
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
        ang_error (float, optional): angular error in deg.
            Defaults to 0.8deg, which is the median value from the 10y NT MC
            (nt_v005_p01)
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
    def compute_opening_angles_per_src(
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
            op_angles[i] = np.arccos(
                np.sin(src_dec) * np.sin(data_dec)
                + np.cos(src_dec) * np.cos(data_dec) * np.cos(src_ra - data_ra)
            )
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

    def create_ts_per_bin(self, n_s: float, data: npt.ArrayLike) -> float:
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

        N = len(data)
        llh_vals = []
        for i, src in enumerate(self.src_pos):
            # compute array per src, each element of the array
            # is the opening angle wrt each event in data
            opening_angles = self.compute_opening_angles_per_src(src, data)
            # compute signal spatial pdf for each src using data
            sig_spatial = self.spatial_pdf(opening_angles)
            print("sig spatial before mask ", sig_spatial)
            sig_spatial_mask = sig_spatial > self.sig_spatial_threshold
            sig_spatial = sig_spatial[sig_spatial_mask]
            print("sig spatial after mask ", sig_spatial)

            # calculate llh value for each src and all the data
            # bkg spatial pdf = 1/4*pi
            llh_value = 1 + (n_s / N) * (4 * np.pi * sig_spatial - 1.0)
            llh_per_src = np.sum(np.log(llh_value))
            llh_vals.append(llh_per_src)

        return 2.0 * np.sum(llh_vals)


srcs = SimulateSources(2)
data = SimulateDataset(src_pop=srcs, smax=5.0, identical=True, n_atm=10, e_bins=3)
print("sig events = ", data.all_sig_events_per_bin)
print("all astro bkg events = ", data.n_astro)
print("expected astro bkg events per bin", data.astro_events_per_bin)
print("expected atm bkg events per bin:", data.atm_events_per_bin)
print("data for 1st bin: ", len(data.dataset[0]))
llh = LLH(data, srcs)
src_pos = srcs.src_positions[0]
print("src position: ", src_pos)
data_pos_first_bin = data.dataset[0]
oa = llh.compute_opening_angles_per_src(src_pos, data_pos_first_bin)
print("opening angles for data in 1st energy bin: ", oa)
print("ang error (rad) = ", llh.ang_err)
spatial_pdf = llh.spatial_pdf(oa)
print("sig pdf vals: ", spatial_pdf)
ts_1st_bin = llh.create_ts_per_bin(1.0, data_pos_first_bin)
print("for ns = 1 & 1st energy bin, TS = ", ts_1st_bin)
