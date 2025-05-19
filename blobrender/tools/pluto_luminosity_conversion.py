import numpy as np
from astropy import constants as const
import astropy.units as u 
kpc = const.kpc.si.value

def lum_unit_si(eta=0.1, gmm1=10, gmm2 = 1e5, c=0.05269, P=1.6e-11, L=2.1 * kpc, q=2.2, nu=(178 * 1e6)):
	'''
	Calculates the luminosity unit in W/Hz/sr given a unit pressure and length in SI units, and with various assumptions
	about the electron spectrum and departure from equipartition.

	equation on page 188 of Hardcastle and Krause 2013: 
	https://ui.adsabs.harvard.edu/abs/2013MNRAS.430..174H/abstract

	Parameters:
		eta 		float
					departure from equipartition  <- eta=3/4 corresponds to minimum energy
		gmm1,gmm2 	float
					min and max Lorentz factors of electron distribution
		c 			float
		P 			float 
					SI simulation unit of pressure (get from PLUTO output file and convert)
		L 			float 
					SI simulation unit of length (get from PLUTO output file and convert)
		q 			float 
					particle spectral index 
		nu 			float
					frequency in Hz 

	'''
	# calculate the integral of E^1-q dE. 
	I = get_I(q, gmm1, gmm2)

	# the various terms that go into the expression separated out a bit arbitrarily
	const1 = c * (const.e.si.value ** 3) / const.eps0.si.value / const.c.si.value / const.m_e.si.value 
	term1 = (nu * (const.m_e.si.value **3) * (const.c.si.value**4) / const.e.si.value) ** (-(q-1)/2.0)
	term2 = 3.0 * (P ** ((q + 5) / 4)) / 4.0 / np.pi / I 
	term3 = (6.0 * const.mu0.si.value * eta) ** ((q + 1) / 4)
	term4 = (1.0 + eta) ** (-(q + 5) / 4)
	term5 = L ** 3 
	lum = const1 * term1 * term2 * term3 * term4 * term5 
	return (lum)

def get_lum_unit_cgs(unit_p, unit_l, **lum_unit_kwargs):
	'''
	Calculates the luminosity unit in W/Hz/sr given a unit pressure and length, both in CGS units, 
	and with various assumptions about the electron spectrum and departure from equipartition 

	Parameters:
		unit_p 			float 
						SI simulation unit of pressure (get from PLUTO output file and convert)
		unit_l 			float 
						SI simulation unit of length (get from PLUTO output file and convert)
		lum_unit_kwargs kwargs 
						keyword arguments to pass to lum_unit_si 	

	Returns:
		lum 			float 
						luminosity unit in erg/s/Hz/sr 
	'''
	P = unit_p * u.Unit("Ba") # Baryes (CGS pressure)
	L = unit_l * u.Unit("cm")

	# factor of 1e7 here converts watts to erg/s
	lum = lum_unit_si(P=P.si.value, L=L.si.value, **lum_unit_kwargs) * 1e7 
	
	return (lum)
    

def get_I(q, gmm1, gmm2):
	'''
	calculate the integral of E^1-q dE. 
	'''
	mcsq = const.m_e.si.value * const.c.si.value * const.c.si.value
	E1 = gmm1 * mcsq 
	E2 = gmm2 * mcsq
	if q == 2:
		term1 = np.log(E2/E1)
		prefactor = 1.0
	else:
		prefactor = 1.0 / (2 - q)
		exponent = (2 - q)
		term1 = (E2**exponent - E1**exponent)

	return (term1 * prefactor)