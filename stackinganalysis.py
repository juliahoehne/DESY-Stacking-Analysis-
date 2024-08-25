#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.optimize import curve_fit
import pandas as pd


class BackgroundNeutrinos:
    def __init__(self, threshold1=1e13, threshold2=1e14):
        self.ra_min = 0 
        self.ra_max = 360 
        self.dec_min = -90 
        self.dec_max = 90 
        self.threshold1 = 1e13  # First energy threshold
        self.threshold2 = 1e14  # Second energy threshold

    def power_law_distribution(self, energy_min, energy_max, size, alpha):
        u = np.random.uniform(0, 1, size)
        energies = (energy_max**(alpha+1) - energy_min**(alpha+1)) * u + energy_min**(alpha+1)
        energies = energies**(1 / (alpha+1))
        return energies

    def simulate_events(self, n_sources, energy_range=(1e12, 1e15)):
        # Generate 50,000 events for the energy distribution with spectral index -3.0
        bkg_sources = 50000
        ra_dist1 = np.random.uniform(0, 360, bkg_sources)
        dec_dist1 = np.arcsin(np.random.uniform(-1, 1, bkg_sources)) * 180 / np.pi
        energies_dist1 = self.power_law_distribution(*energy_range, bkg_sources, -3.0)
    
        # Generate 500 events for the energy distribution with spectral index -1.9
        sig_sources = 500
        ra_dist2 = np.random.uniform(self.ra_min, self.ra_max, sig_sources)
        dec_dist2 = np.arcsin(np.random.uniform(-1, 1, sig_sources)) * 180 / np.pi
        energies_dist2 = self.power_law_distribution(*energy_range, sig_sources, -1.9)

        return {
            'atm bkg': (ra_dist1, dec_dist1, energies_dist1),
            'astr bkg': (ra_dist2, dec_dist2, energies_dist2)
        }
    
    def filter_energies(self, energies, ra, dec):
        below_threshold1 = []
        between_thresholds = []
        above_threshold2 = []

        for energy, ra_val, dec_val in zip(energies, ra, dec):
            if energy <= self.threshold1:
                below_threshold1.append((energy, ra_val, dec_val))
            elif energy <= self.threshold2:
                between_thresholds.append((energy, ra_val, dec_val))
            else:
                above_threshold2.append((energy, ra_val, dec_val))

        return below_threshold1, between_thresholds, above_threshold2


        background_neutrinos = BackgroundNeutrinos(threshold1=1e13, threshold2=1e14)


        simulation_results = background_neutrinos.simulate_events(n_sources=50000)


        ra_atm_bkg, dec_atm_bkg, energies_atm_bkg = simulation_results['atm bkg']
        ra_astr_bkg, dec_astr_bkg, energies_astr_bkg = simulation_results['astr bkg']


        energies_atm_below, energies_atm_between, energies_atm_above = background_neutrinos.filter_energies(energies_atm_bkg, ra_atm_bkg, dec_atm_bkg)
        energies_astr_below, energies_astr_between, energies_astr_above = background_neutrinos.filter_energies(energies_astr_bkg, ra_astr_bkg, dec_astr_bkg)


    def extract_coordinates_and_energies(filtered_energies):
        ra = [entry[1] for entry in filtered_energies]
        dec = [entry[2] for entry in filtered_energies]
        energies = [entry[0] for entry in filtered_energies]
        return ra, dec, energies

# Extract RA, Dec, and energies for each category
        ra_atm_below, dec_atm_below, energies_atm_below = extract_coordinates_and_energies(energies_atm_below)
        ra_atm_between, dec_atm_between, energies_atm_between = extract_coordinates_and_energies(energies_atm_between)
        ra_atm_above, dec_atm_above, energies_atm_above = extract_coordinates_and_energies(energies_atm_above)

        ra_astr_below, dec_astr_below, energies_astr_below = extract_coordinates_and_energies(energies_astr_below)
        ra_astr_between, dec_astr_between, energies_astr_between = extract_coordinates_and_energies(energies_astr_between)
        ra_astr_above, dec_astr_above, energies_astr_above = extract_coordinates_and_energies(energies_astr_above)

# Plotting the results
        plt.figure(figsize=(15, 10))

# Atmospheric background neutrinos
        plt.subplot(2, 3, 1)
        plt.scatter(ra_atm_below, dec_atm_below, c=energies_atm_below, cmap='viridis', s=1)
        plt.colorbar(label='Energy')
        plt.title('Atmospheric Background Neutrinos Below 1e13eV')
        plt.xlabel('RA')
        plt.ylabel('Dec')

        plt.subplot(2, 3, 2)
        plt.scatter(ra_atm_between, dec_atm_between, c=energies_atm_between, cmap='viridis', s=1)
        plt.colorbar(label='Energy')
        plt.title('Atmospheric Background Neutrinos Between 1e13eV and 1e14eV')
        plt.xlabel('RA')
        plt.ylabel('Dec')

        plt.subplot(2, 3, 3)
        plt.scatter(ra_atm_above, dec_atm_above, c=energies_atm_above, cmap='viridis', s=1)
        plt.colorbar(label='Energy')
        plt.title('Atmospheric Background Neutrinos Above 1e14eV')
        plt.xlabel('RA')
        plt.ylabel('Dec')

# Astrophysical background neutrinos
        plt.subplot(2, 3, 4)
        plt.scatter(ra_astr_below, dec_astr_below, c=energies_astr_below, cmap='viridis', s=1)
        plt.colorbar(label='Energy')
        plt.title('Astrophysical Background Neutrinos Below 1e13eV')
        plt.xlabel('RA')
        plt.ylabel('Dec')

        plt.subplot(2, 3, 5)
        plt.scatter(ra_astr_between, dec_astr_between, c=energies_astr_between, cmap='viridis', s=1)
        plt.colorbar(label='Energy')
        plt.title('Astrophysical Background Neutrinos Between 1e13eV and 1e14eV')
        plt.xlabel('RA')
        plt.ylabel('Dec')

        plt.subplot(2, 3, 6)
        plt.scatter(ra_astr_above, dec_astr_above, c=energies_astr_above, cmap='viridis', s=1)
        plt.colorbar(label='Energy')
        plt.title('Astrophysical Background Neutrinos Above 1e14eV')
        plt.xlabel('RA')
        plt.ylabel('Dec')

        plt.tight_layout()
        plt.show()


    def plot_histograms(self, events):
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        fig.subplots_adjust(hspace=0.3)

        for i, (event_type, (ra, dec, energies)) in enumerate(events.items()):
            ra = np.array(ra)
            dec = np.array(dec)
            energies = np.array(energies)
            
            # Plot histogram for RA
            ax[0, i].hist(ra, bins=60, color='blue', alpha=0.5)
            ax[0, i].set_xlabel('Right Ascension (RA)')
            # ax[0, i].set_ylabel('Number of Sources')
            ax[0, i].set_title(f'Histogram of Right Ascension (RA) for {event_type.capitalize()}')
            ax[0, i].grid(True)
            
            # Plot histogram for Dec
            ax[1, i].hist(np.sin(dec*np.pi/180), bins=60, color='green', alpha=0.5)
            ax[1, i].set_xlabel('Declination (Dec)')
            # ax[1, i].set_ylabel('Number of Sources')
            ax[1, i].set_title(f'Histogram of Declination for {event_type.capitalize()}')
            ax[1, i].grid(True)

            # Plot histogram for energies
            ax[i, 2].hist(np.log10(energies), bins=60, color='red', alpha=0.5, label=f'{event_type.capitalize()}')
            ax[i, 2].set_xlabel('log10(Energy)')
            # ax[i, 2].set_ylabel('Number of Sources')
            ax[i, 2].set_title('Histogram of Energy')
            ax[i, 2].semilogy()
            ax[i, 2].grid(True)
            ax[i, 2].legend()

        fig.savefig('Histograms_Background.png')

        # Combined histogram for energies
        plt.figure(figsize=(10, 5))
        plt.hist([np.log10(events['atm bkg'][2]), np.log10(events['astr bkg'][2])], bins=60, color=['blue', 'red'], alpha=0.5, label=['Atm Bkg', 'Astr Bkg'])
        plt.xlabel('log10(Energy) ')
        plt.ylabel('Number of Sources')
        plt.title('Combined Histogram of Background Energy')
        plt.semilogy()
        plt.grid(True)
        plt.legend()
        plt.savefig("Background_Energy.png")
        plt.show()

    def plot_skymap(self, events, nside=64):
        for event_type, (ra, dec, energies) in events.items():
            # Convert RA and Dec to theta and phi for healpy
            theta = np.deg2rad(90 - dec)
            phi = np.deg2rad(ra)
            
            # Create a healpy map
            skymap = np.zeros(hp.nside2npix(nside))
            pixels = hp.ang2pix(nside, theta, phi)
            for pix in pixels:
                skymap[pix] += 1
            
            # Plot the healpy map
            hp.mollview(skymap, title=f"Skymap of {event_type.capitalize()} Sources for background", cmap="viridis")
            hp.graticule()
            plt.show()
            

class SignalNeutrinos:
    
    def __init__(self, smax, nsources, spectral_index=-1.9, dnds_index=-2.5, verbose=False):
        self.Smax = smax
        self.nSources = nsources
        self.spectralIndex = spectral_index
        self.dNdSIndex = dnds_index
        
        self.sN = lambda n: self.Smax * n**(1./(self.dNdSIndex + 1.))
        self.Smin = self.sN(nsources)
        
        if verbose:
            print(f"smax: {self.Smax}")
            print(f"nsources: {self.nSources}")
            print(f"Source population:")
            print(f"   Min flux: {self.Smin}")
            print(f"   Max flux: {self.Smax}")
            total_flux = (self.dNdSIndex + 1.) / (self.dNdSIndex + 2.) * self.Smax * (self.nSources**((self.dNdSIndex + 2.) / (self.dNdSIndex + 1.)))
            print(f"   Total flux: {total_flux}")
    
    def createSample(self, lebins=None, invisible_fraction=0., identity_sources=False, use_poisson=True):
        if identity_sources:
            self.srcs = self.nSources * np.ones(self.nSources)
        else:
            self.srcs = self.nSources * np.random.rand(self.nSources)
        
        self.totalFlux = self.sN(self.srcs)
        
        self.isVisible = np.ones(self.nSources)
        if invisible_fraction > 0.:
            self.isVisible = np.where(np.random.rand(self.nSources) < invisible_fraction, 0., 1.)
        
        self.nSourcesVisible = int(self.isVisible.sum())
        print(f"Sources: {self.nSources}, Visible: {self.nSourcesVisible}")

        # Generate isotropic distribution of sources
        self.ra_dist = np.random.uniform(0, 360, self.nSources)
        self.dec_dist = np.arcsin(np.random.uniform(-1, 1, self.nSources)) * 180 / np.pi
        
        self.saveDistributionsToCSV()

        if lebins is not None:
            self.fluxPerBin=np.zeros((self.nSources,len(lebins)-1)) 
            self.nEventsPerBin=np.zeros((self.nSources,len(lebins)-1)) 
            g1=self.spectralIndex+1
            phi0=g1*self.totalFlux/( 10**(lebins[-1]*g1) - 10**(lebins[0]*g1) )
            
            for i,le in enumerate(lebins[:-1]):
                self.fluxPerBin[:,i]=phi0/g1*(10**(lebins[i+1]*g1) - 10**(lebins[i]*g1))
                self.nEventsPerBin[:,i]=np.random.poisson(self.fluxPerBin[:,i])*self.isVisible #i can change this to if poisson. Make poisson default and if else we keep it the same meaning we just keep flux per bin
                self.fluxPerBin[:,i]*=self.isVisible
                
            self.nEventsTotal=self.nEventsPerBin.sum(1)
           
        else:
            self.fluxPerBin=None
            self.nEventsPerBin=None
            self.nEventsTotal=np.random.poisson(self.totalFlux)*self.isVisible
            
    def saveDistributionsToCSV(self, filename="signal_neutrinos_distribution.csv"):
        data = {
            "ra_dist": self.ra_dist,
            "dec_dist": self.dec_dist
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Distributions saved to {filename}")
    
    def plot_ra_dec_distribution(self):
        """Create plot for right ascension and declination"""
        #RA
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(self.ra_dist, bins=50, color='red', alpha=0.5)
        plt.xlabel('RA (degrees)')            
        plt.grid(True)                             
        plt.title('RA Distribution')
        
        #DEC
        plt.subplot(1, 2, 2)
        plt.hist(np.sin(self.dec_dist*np.pi/180), bins=50, color='blue', alpha=0.5)
        plt.xlabel('Dec (degrees)')
        plt.grid(True)                             
        plt.title('Dec Distribution')
        
        plt.tight_layout()
        plt.savefig("RA and DEC for signal")
        plt.show()
    
    def plot_skymap(self, nside=64):
        """Create skymap of sources to check if they are isotropically distributed. We use healpy method. Modify nside number to change the resolution of the map. """
        # Convert RA and Dec to theta and phi for healpy
        theta = np.deg2rad(90 - self.dec_dist)
        phi = np.deg2rad(self.ra_dist)
        
        skymap = np.zeros(hp.nside2npix(nside))
        pixels = hp.ang2pix(nside, theta, phi)
        for pix in pixels:
            skymap[pix] += 1
        
        hp.mollview(skymap, title="Skymap of Sources", cmap="viridis")
        hp.graticule()
        plt.savefig("Skymap Signal Sources")
        plt.show()

    def plot_total_flux(self):
        """Create plot for total flux, using fit curve from scipy. Fit S^-5/2 to check if the plot is correct.
        -Bin Centers: Used as x-values for fitting because they represent the midpoints of the histogram bins.
        -Curve Fit: `curve_fit` function optimizes the value of \( A \) to fit the histogram data."""
        
        plt.figure(figsize=(10,5))
        hist, bins, _= plt.hist(self.totalFlux, bins=50, color= "blue", alpha=0.7, label="Total Flux")
        bin_centers = (bins[:-1] + bins[1:])/2

        #Define S^-5/2 power law with normalisation
        def power_law(x,A):
            return A * x**(-5/2)
        popt,_ = curve_fit(power_law,bin_centers,hist,maxfev=1000)

        plt.plot(bin_centers, power_law(bin_centers,*popt), "r-", label=r'Fit: $S^{-5/2}$')

        plt.xlabel('Total Flux')
        plt.ylabel('Number of Sources')
        plt.title('Total Flux Distribution with $S^{-5/2}$ fit curve')
        plt.legend()
        plt.grid(True)
        plt.savefig("Total Flux Distribution")
        plt.show()

    def plot_flux_per_energy(self, lebins, normalize=True):
            """Plots the flux per bin with energy from range 3 to 6 GeV"""
            self.createSample(lebins=lebins,invisible_fraction=0.1, identity_sources=False)
            flux_per_energy_bin_random = self.fluxPerBin
            self.createSample(lebins=lebins, invisible_fraction=0.1, identity_sources=True)
            flux_per_energy_bin_identical = self.fluxPerBin
            
            # Summing flux per energy bin for all sources
            total_flux_per_energy_bin_random = flux_per_energy_bin_random.sum(axis=0)
            total_flux_per_energy_bin_identical = flux_per_energy_bin_identical.sum(axis=0)

            if normalize:
                total_flux_per_energy_bin_random /= self.nSources
                total_flux_per_energy_bin_identical /= self.nSources

            energy_bins = 10**lebins
            
            plt.plot(energy_bins[:-1], total_flux_per_energy_bin_random, 'b-', alpha=0.7, label='Random Sources')
            plt.plot(energy_bins[:-1], total_flux_per_energy_bin_identical, 'r-', alpha=0.7, label='Identical Sources')
            
            plt.xlabel('Energy (GeV)')
            plt.ylabel('Total Flux per # of Sources')
            plt.title('Total Flux for Random and Identical Sources')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.show()

    def plot_neutrinos_per_energy(self, lebins):
        # Create samples for random and identical sources
        self.createSample(lebins=lebins, invisible_fraction=0.1, identity_sources=False, use_poisson=True)
        random_with_poisson = self.nEventsPerBin
        self.createSample(lebins=lebins, invisible_fraction=0.1, identity_sources=True, use_poisson=True)
        identical_with_poisson = self.nEventsPerBin
        
        # Sum neutrinos per energy bin for all sources
        sum_random_with_poisson = random_with_poisson.sum(axis=0)
        sum_identical_with_poisson = identical_with_poisson.sum(axis=0)

        #Normalised 
        sum_random_with_poisson /= self.nSources
        sum_identical_with_poisson /= self.nSources

        plt.figure(figsize=(10, 6))
        energy_bins = 10**lebins
        
        plt.plot(energy_bins[:-1], sum_random_with_poisson, 'b-', alpha=0.7, label='Random Sources')
        plt.plot(energy_bins[:-1], sum_identical_with_poisson, 'r-', alpha=0.7, label='Identical Sources') #with poisson
        
        plt.xlabel('Energy (GeV)')
        plt.ylabel('Number of Neutrinos per Energy Bin')
        plt.title('Number of Neutrinos per Energy Bin for Different Cases')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    simulator = BackgroundNeutrinos()
    n_sources = 5000
    events = simulator.simulate_events(n_sources)
    
    # Plot histograms
    simulator.plot_histograms(events)
    
    # Plot skymap
    simulator.plot_skymap(events)

lebins = np.log10(np.linspace(3, 6, 6))
signal_random = SignalNeutrinos(smax=5, nsources=500, verbose=True)
signal_random.createSample(lebins, invisible_fraction=0.1, identity_sources=False, use_poisson=True)

#Plots
signal_random.plot_ra_dec_distribution()
signal_random.plot_skymap()
signal_random.plot_total_flux()
signal_random.plot_flux_per_energy(lebins)
signal_random.plot_neutrinos_per_energy(lebins)




# In[76]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

class DataSimulation:
    def __init__(self, sigma=0.8, spatial_box_width=5.0, mean_dec=0.73, std_dec=0.47):
        self.sigma = sigma
        self.spatial_box_width = spatial_box_width
        self.mean_dec = mean_dec
        self.std_dec = std_dec
    

    def simulate_distribution(self, data):
        data = pd.DataFrame(data)
        data["dec"] = norm.rvs(loc=self.mean_dec, scale=self.std_dec, size=len(data))
        cos_dec = np.cos(data["dec"])
        non_zero_mask = cos_dec != 0
        
        data["ra"] = np.nan
        data.loc[non_zero_mask, "ra"] = (np.pi + norm.rvs(size=non_zero_mask.sum()) * self.sigma) / cos_dec[non_zero_mask]
        data["sinDec"] = np.sin(data["dec"])
        
        return data.copy()

    def select_spatially_coincident_data(self, data, sources):
        """Checks each source, and only identifies events in data which are
        both spatially and time-coincident with the source. Spatial
        coincidence is defined as a +/- 5 degree box centered on the given
        source. Time coincidence is determined by the parameters of the LLH
        Time PDF. Produces a mask for the dataset, which removes all events
        which are not coincident with at least one source.

        :param data: Dataset to be tested
        :param sources: Sources to be tested
        :return: Mask to remove
        """
        veto = np.ones_like(data["ra"], dtype=bool)

        for source in sources:
            width = np.deg2rad(self.spatial_box_width)
            min_dec = max(-np.pi / 2.0, source["dec_rad"] - width)
            max_dec = min(np.pi / 2.0, source["dec_rad"] + width)
            
            dec_mask = np.logical_and(np.greater(data["dec"], min_dec), np.less(data["dec"], max_dec))
            cos_factor = np.amin(np.cos([min_dec, max_dec]))
            dPhi = np.amin([2.0 * np.pi, 2.0 * width / cos_factor]) 
            
            ra_dist = np.fabs((data["ra"] - source["ra_rad"] + np.pi) % (2.0 * np.pi) - np.pi)
            ra_mask = ra_dist < dPhi / 2.0

            spatial_mask = dec_mask & ra_mask
            veto = veto & ~spatial_mask

        return ~veto

    @staticmethod
    def angular_distance(x1, x2):
        """
        Compute the angular distance between 2 vectors on a unit sphere
        Parameters :
            x1/2: Vector with [declination, right-ascension], i.e.
                  shape (2,n) where n can also be zero. One can compute
                  the distance for n pairs of vectors
        Return :
            angular distance of len(xi)
        """
        x1 = np.array(x1, dtype=np.float64)
        x2 = np.array(x2, dtype=np.float64) 
        assert len(x1) == len(x2) == 2
        return np.arccos(np.sin(x1[0]) * np.sin(x2[0]) \
                + np.cos(x1[0]) * np.cos(x2[0]) * np.cos(x1[1]-x2[1]))

if __name__ == "__main__":
    sim = DataSimulation(sigma=0.8)
    num_events = 100
    dtype = [("sigma", float), ("ra", float), ("dec", float), ("sinDec", float)]
    data = np.zeros(num_events, dtype=dtype)
    data["sigma"] = 0.8
    simulated_data = sim.simulate_distribution(data)

    # Define sources (for example, one source at dec=0.73 and ra=np.pi)
    sources = [{"dec_rad": 0.73, "ra_rad": np.pi}]

    # Apply spatial coincidence check
    coincidence_mask = sim.select_spatially_coincident_data(simulated_data, sources)
    
    # Filter coincident events
    coincident_data = simulated_data[coincidence_mask]
    num_coincident_neutrinos = len(coincident_data)

    print(f"Number of coincident neutrinos: {num_coincident_neutrinos}")

    if num_coincident_neutrinos > 0:
        # Calculate angular distances for coincident neutrinos
        true_coords = np.array([0, np.pi])  # trueDec=0, trueRa=np.pi
        coincident_event_coords = np.vstack((coincident_data["dec"], coincident_data["ra"]))
        angular_distances = sim.angular_distance(coincident_event_coords, true_coords[:, None])

        # Plot histogram of angular distances
        plt.figure(figsize=(8, 6))
        plt.hist(angular_distances, bins=50, alpha=0.75)
        plt.xlabel('Angular Distance [°]')
        plt.ylabel('Frequency')
        plt.title('Histogram of Angular Distance for Coincident Neutrinos')
        plt.grid(True)
        plt.show()

        # Calculate spatial PDF for coincident neutrinos
        def spatial_pdf(r, sigma):
            return (1 / (2 * np.pi * sigma**2)) * np.exp(-r**2 / (2 * sigma**2))

        sigma = 0.8  # rad
        spatial_pdf_values = spatial_pdf(angular_distances, sigma)

        # Plot histogram of spatial PDF
        plt.figure(figsize=(8, 6))
        plt.hist(angular_distances, bins=50, weights=spatial_pdf_values, alpha=0.75)
        plt.xlabel('Angular Distance [°]')
        plt.ylabel('Spatial PDF')
        plt.title(f'Spatial PDF Histogram for Coincident Neutrinos (Sigma = {sigma})')
        plt.grid(True)
        plt.show()
    else:
        print("No coincident neutrinos found.")


# In[81]:


import pandas as pd
import numpy as np

class llh:
    def __init__(self, lifetime=10, sigma=0.8, N=50500):
        self.lifetime = lifetime
        self.sigma = sigma
        self.N = N    # n_s(poisson) + n_bkg(poisson)
        self.data_simulator = DataSimulation(sigma=sigma)

    def llh_B(self):
        l_btime = 1 / self.lifetime
        l_bspatial = 1 / (4 * np.pi)
        l_B = self.N * l_btime * l_bspatial  # n_bkg=50500
    
        print(f"l_B (Background Likelihood): {l_B}")
    
        return l_B


    def spatial_pdf(self, r):
        return (1 / (2 * np.pi * self.sigma**2)) * np.exp(-r**2 / (2 * self.sigma**2))

    def llh_S(self, angular_distances):
        l_stime = 1 / self.lifetime
        l_sspatial = self.spatial_pdf(angular_distances)
        l_S = l_stime * l_sspatial
        return l_S

    def llh_hypothese(self, angular_distances, n_s=10):
        l_B = self.llh_B()
        l_S = self.llh_S(angular_distances)
        l_hypo = (n_s / self.N) * l_S + ((self.N - n_s) / self.N) * l_B  # n_s from spatial coincidence
    
        print(f"l_B (Background Likelihood): {l_B}")
        print(f"l_S (Signal Likelihood): {l_S}")
        print(f"l_hypo (Combined Likelihood): {l_hypo}")
    
        return l_hypo


    def TS(self, combined_likelihood, l_B):
        product_l_hypo = np.prod(combined_likelihood)  # product of every llh
        test_stat = product_l_hypo / l_B
        log_test_stat = 2 * np.log(test_stat)  # log
        return log_test_stat

# CSV file of sources
df = pd.read_csv('signal_neutrinos_distribution.csv')

# Simulation of the distribution
sim = DataSimulation(sigma=0.8)
num_events = 1000
dtype = [("sigma", float), ("ra", float), ("dec", float), ("sinDec", float)]
data = np.zeros(num_events, dtype=dtype)
data["sigma"] = 0.8
simulated_data = sim.simulate_distribution(data)

# Likelihood
likelihood_calculator = llh(lifetime=10, sigma=0.8, N=50500)
l_B = likelihood_calculator.llh_B()

# Test statistics for each source
TS_values = []

# Loop over all sources
for index, row in df.iterrows():
    new_dec = row['dec_dist']
    new_ra = row['ra_dist']

    # true_coords are the coordinates of the source
    true_coords = np.array([new_dec, new_ra])

    # Angular distance
    event_coords = np.vstack((simulated_data["dec"], simulated_data["ra"]))
    angular_distances = sim.angular_distance(event_coords, true_coords[:, None])

    # Spatial coincidence check for this source
    source = {"dec_rad": new_dec, "ra_rad": new_ra}
    coincidence_mask = sim.select_spatially_coincident_data(simulated_data, [source])
    
    # Calculate n_s: number of coincident neutrinos
    n_s = np.sum(coincidence_mask)

    # Likelihood value for the coincident neutrinos
    if n_s > 0:
        coincident_angular_distances = angular_distances[coincidence_mask]
        l_hypo = likelihood_calculator.llh_hypothese(coincident_angular_distances, n_s)
        
        # Combine the likelihoods for this source
        combined_likelihood = np.prod(l_hypo)  # Product of likelihoods for this source

        # Calculate the Test Statistic (TS) for this source
        TS_value = likelihood_calculator.TS(combined_likelihood, l_B)
        TS_values.append(TS_value)
    else:
        TS_values.append(0)  # No coincident neutrinos

# Output TS values for all sources
print("TS values for all sources:", TS_values)

non_zero_TS_values = [ts for ts in TS_values if ts != 0]

# Erstelle ein Histogramm
plt.figure(figsize=(10, 6))
plt.hist(non_zero_TS_values, bins=30, edgecolor='black', alpha=0.7)

# Titel und Achsenbeschriftungen
plt.title('Histogram of TS values (non-zero)', fontsize=16)
plt.xlabel('TS Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Zeige das Histogramm an
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




