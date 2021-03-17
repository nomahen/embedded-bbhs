'''
agn.py contains information for produces interpolated AGN profiles based on a certain AGN model.
When you run this, it loads an opacity table used for Sirko-Goodman, you'll want to modify the paths here.
To make a new model, add a new instance of the AGN_model class; this calculates properties at a specific
distance, and AGN_profile will loop through a distance array to construct the profile. AGN_profile has attributes
associated with every AGN model (SMBH Mass, Distance in AGN disk, Eddington Ratio, etc), and there are additional
attributes that are specific to an AGN model (i.e., SG model takes the parameter 'b' that dictates whether
viscosity is proportional to gas or gas+radiation pressure), you pass these to AGN_profile in the 'p' array,
(so, if you want an AGN_profile using the model sirko_goodman, you set p = np.array([b, tau_guess, pgas_guess])).
If youre AGN model has no extra parameters, this can just be an empty array.
'''

import numpy as np
from scipy.interpolate import interp2d, interp1d
from scipy.optimize import least_squares
from scipy.integrate import quad, solve_ivp

## get constants
from constants import *

## functions

def lambda_s(g):
    """ this is lambda_s parameter used in Bondi accretion, see Shapiro + Teukolsky. g here is adiabatic index."""
    return (0.5)**((g + 1.)/(2*(g - 1.))) * ((5. - 3.*g)/4.)**(-1.*(5. - 3.*g)/(2.*(g-1.)))

def init_opacity():
    '''
    This interpolates data from two opacity tables for the sirko-goodman profile. It's initialized at the end of this
    script.

    Modify paths for your computer
    '''

    ### Low T

    ## interpolate from opacity table for sirko goodman profiles
    path_to_table = '/home/nkaaz/research/data/opacity_tables/f05_caffau/caffau11.7.06.tron'
    table = np.genfromtxt(path_to_table,skip_header=4)
    # logR is a top row
    logR  = np.genfromtxt(path_to_table,skip_header=3,skip_footer=85)[2:]
    logT  = table[:,0]
    low_logTmin, low_logTmax = np.min(logT), np.max(logT)
    low_logRmin, low_logRmax = np.min(logR), np.max(logR)
    opacities = table[:,1:]
    opacity_function_lowT = interp2d(logR,logT,opacities,bounds_error=False)

    ### High T

    ## interpolate from opacity table for sirko goodman profiles
    path_to_table = '/home/nkaaz/research/data/opacity_tables/table76'
    table = np.genfromtxt(path_to_table,skip_header=6,invalid_raise=False, missing_values = "", filling_values=0.0)
    # logR is a top row
    logR  = np.genfromtxt(path_to_table,skip_header=4,skip_footer=70)[1:]
    logT  = table[:,0]
    high_logTmin, high_logTmax = np.min(logT), np.max(logT)
    high_logRmin, high_logRmax = np.min(logR), np.max(logR)
    opacities = table[:,1:]
    opacity_function_highT = interp2d(logR,logT,opacities,bounds_error=False)

    def opacity_function_allT(R,T,Tcrit = 8000):
        # Disk should not exceed temperature of Tmax in opacity table
        if T <= Tcrit:
            return opacity_function_lowT(np.log10(R),np.log10(T))
        else:
            return opacity_function_highT(np.log10(R),np.log10(T))

        # Check if below Tmin. Follow SG approach
        if T <= 10.**low_logTmin: # SG 2003
            print("below T")
            return 10.**(0.76) # cm^2/g

        # Check if below Rmin. Rmin/max are same for both tables.
        if (R <= 10.**low_logRmin):
            print("below R")
            if T <= Tcrit:
                return opacity_function_lowT(low_logRmin,np.log10(T))
            else:
                return opacity_function_highT(low_logRmin,np.log10(T))

        # Check if above Rmax. Rmin/max are same for both tables.
        if (R >= 10.**low_logRmax):
            print("above R")
            if T <= Tcrit:
                return opacity_function_lowT(low_logRmax,np.log10(T))
            else:
                return opacity_function_highT(low_logRmax,np.log10(T))


    opacity_function_allT_vec = np.vectorize(opacity_function_allT)


    return opacity_function_allT_vec

# classes

class binary:
    ''' equal mass binary '''

    def __init__(self, Mbbh, sma, RadEff=1e-2):
        '''
        Mbbh ~ BBH mass in solar masses
        sma  ~ BBH semimajor axis in au
        RadEff ~ radiative efficiency of BBH accreting from AGN disk
        '''
        self.Mbbh = Mbbh # BBH mass in solar masses
        self.sma  = sma  # BBH semimajor axis in au
        self.L    = np.sqrt(G/16.)*(self.Mbbh*smass_to_g)**(1.5)*(self.sma*au_to_cm)**(0.5) # Angular momentum of BBH in cgs

        # Eddington accretion rate, assuming 10% radiative efficiency
        self.Edd_Mdot = self._Edd_Mdot(self.Mbbh,RadEff=RadEff)#1.26e38 * self.Mbbh/(c*c*RadEff)

    @staticmethod
    def _Edd_Mdot(M,RadEff=1e-2):
        '''
        Calculate the Eddington-limited accretion rate

        Parameters
        M      ~ mass in solar masses
        RadEff ~ radiative efficiency

        Returns
        Mdot_Edd ~ Eddington accretion rate, grams/second
        '''

        # M in solar msses
        return 1.26e38*M/(c*c*RadEff)

    @staticmethod
    def _t_gw(Mbbh,sma):
        '''
        returns gravitational wave timescales

        Mbbh in solar masses,
        sma  in au
        '''
        sma  *= au_to_cm   # convert from au to centimeters
        Mbbh *= smass_to_g # convert from solar mass to grams

        # return in seconds
        return (5./64.) * (c**5./G**3.) * (sma**4.) / Mbbh**3.



class AGN_profile:
    '''
    Creates profile of quantities for an AGN model
    '''
    def __init__(self,which_model,Msmbh,D_arr,alpha,EddRatio,RadEff,p,BBH=None,verbose=1,gen=1, interp=1):
        '''
        Parameters

        which_model      instance of AGN_model class to base profiles on
        Msmbh            supermassive black hole mass (solar masses)
        D_arr            array of distances to compute profile for (pc)
        alpha            alpha viscosity parameter (<1)
        EddRatio         Eddington Ratio of AGN disk

        p                Parameters that are specific to an AGN disk model; see comments for chosen AGN disk model (list)

        BBH              instance of binary class to include an embedded binary. optional

        verbose          whether or not to output most print statements (boolean)
        gen              whether or not to generate profiles automatically (boolean)
        interp           whether or not to interpolate profiles automatically. can only be true if gen = 1 (boolean)
        '''


        self.model    = which_model
        self.Msmbh    = Msmbh       # in solar masses
        self.D_arr    = D_arr       # array in distances in pc
        self.alpha    = alpha       # alpha viscosity parameter
        self.EddRatio = EddRatio    # desired Eddington Ratio
        self.RadEff   = RadEff      # Radiative efficiency
        self.Mdot     = self.EddRatio * (1.26e38)*(self.Msmbh) / c**2. / self.RadEff # convert from Eddington ratio to grams/sec
        self.p        = p           # Array of non-standard parameters specific to AGN disk models
        self.N        = len(D_arr)  # array size of profile

        ## extra arguments
        self.verbose = verbose      # turns on/off most print statements

        # Optionally initialize attributes related to embedded binary
        # Profile quantities related to BBH are only generated if these attributes are initialized
        if BBH != None: self._init_bbh(BBH)

        # Calculate AGN parameters as a function of D
        if gen == 1:
            self._gen_profiles()
            self._gen_migration_profiles()
            if BBH != None: self._gen_bbh_profiles()

        # Interpolate AGN parameters as a function of D
        if interp == 1:
            if gen == 1:
                self._interp_profiles_all()
            else:
                print("Can't interpolate unless you've generated profiles; set gen=1!!!")

    # initialize properties of embedded BBH
    def _init_bbh(self,BBH,gamma=1.1):
        '''
        Initialize attributes for an embedded BBH

        Parameters

        BBH    ~ instance of binary class
        gamma  ~ adiabatic index of accretion flow surrounding embedded BBH
        '''


        self.BBH   = BBH
        self.Mbbh  = BBH.Mbbh # BBH mass in solar masses
        self.abbh  = BBH.sma  # BBH semimajor axis in au
        self.L     = BBH.L    # BBH angular momentum in cgs
        self.q     = self.Mbbh/self.Msmbh # BBH to SMBH mass ratio
        self.g     = gamma    # gamma is the adiabatic index for accretion flow onto BBH

    # generate profiles
    def _gen_profiles(self):
        '''
        At each distance D in self.D_arr, generate the AGN model there, append attributes to dictionary
        '''

        ## Initialize dictionary
        self.profiles = {}

        ## Iterate through distance array
        for i,D in enumerate(self.D_arr):
            if self.verbose: print("%4.2f %% done..." % (100.*(1.0*i)/self.N))

            # calculate profile at this distance
            model_at_D = self.model(self.Msmbh,D,self.alpha,self.EddRatio,self.RadEff,*list(self.p))

            # some calculations are done after profile is calculated
            model_at_D._calc()

            # self.p may include 'guesses' which are useful to update
            if hasattr(model_at_D,"succeed"):
                if model_at_D.succeed: self.p = np.copy(model_at_D.p())
            else:
                self.p = np.copy(model_at_D.p())

            # first iteration, initialize profile
            if i == 0:
                for attr in model_at_D.__dict__:
                    self.profiles[attr] = np.array([model_at_D.__dict__[attr]])
            else:
                for attr in model_at_D.__dict__:
                    self.profiles[attr] = np.concatenate((self.profiles[attr],np.array([model_at_D.__dict__[attr]])))

    def _gen_migration_profiles(self):
        '''
        Generates profile quantities that are specific to migration

        These depend on already existing profiles
        '''

        # Eq 3,4 Secunda 2019
        self.profiles["torque_alpha"] = -1.*np.gradient(np.log(self.profiles["sigma"]))/np.gradient(np.log(self.D_arr*pc_to_cm))
        self.profiles["torque_beta"]  = -1.*np.gradient(np.log(self.profiles["T"]))/np.gradient(np.log(self.D_arr*pc_to_cm))
        # disk mass ratio, ie Paardekooper 2014
        self.profiles["qd"]      = np.pi*(self.D_arr*pc_to_cm)**2. * self.profiles["sigma"]/self.Msmbh/smass_to_g
        # migration timescale, ie Paardekooper 2014
        if hasattr(self,"BBH"):
            self.profiles["t_migr"]  = (np.pi/2.)*self.profiles["H_R"]**2./self.profiles["qd"]/self.q/self.profiles["omega"]
            # migration rate, Eq 1 of Secunda 2019
            # assumes isothermal disk
            self.profiles["Ddot_D"] = (-0.85-self.profiles["torque_alpha"]-0.9*self.profiles["torque_beta"])/self.profiles["t_migr"]

    # calculate profiles if a BBH is embedded
    def _gen_bbh_profiles(self):
        '''
        Calculate profile quantities specific to an embedded binary
        '''

        # Check that a BBH has been initialized for this profile
        assert hasattr(self,"BBH")

        # Bondi radius
        self.profiles["Rb"]      = self.Mbbh*smass_to_g*G/self.profiles["cs"]**2.
        # mach H parameter
        self.profiles["mach"]    = (0.5*self.q)**(1./3.) * (self.profiles["H_R"])**(-1.)


        # convenience functions
        # m ~ mach_w
        machfunc1 = lambda m: (m**3./(8.+4*m**2.) + 1)
        machfunc2 = lambda m: (1. + 0.5*m**2.)


        # accretion related quantities
        self.profiles["Rbw"]     = self.profiles["Rb"]/machfunc2(self.profiles["mach"])

        self.profiles["Mdot"]    = 4.*np.pi*self.profiles["Rbw"]**2.*self.profiles["rho"]*self.profiles["cs"]*machfunc1(self.profiles["mach"])
        self.profiles["Ldot"]    = (np.pi/6.)*self.profiles["rho"]*self.profiles["cs"]**2.*self.profiles["Rbw"]**3. * machfunc2(self.profiles["mach"])**(-2.)
        self.profiles["adot_a_inf"] = 2.*self.profiles["Ldot"]/self.L - self.profiles["Mdot"]/(self.Mbbh*smass_to_g)

        lam = lambda_s(self.g)
        self.profiles["adot_a_drg"] = -1. * (self.profiles["Mdot"]/(self.Mbbh*smass_to_g))*(8**0.5)*lam*machfunc2(self.profiles["mach"])**(0.5)*machfunc1(self.profiles["mach"])**(-1.)*(1. + 4.*self.abbh*au_to_cm/self.profiles["Rb"])**(-0.5)

        # timescales
        self.profiles["t_insp_inf"]  = -1./self.profiles["adot_a_inf"]
        self.profiles["t_insp_drg"]  = -1./self.profiles["adot_a_drg"]
        self.profiles["t_insp_all"] = -1./(self.profiles["adot_a_inf"] + self.profiles["adot_a_drg"])

        # accretion efficiency for modified versions
        lkep = self.profiles["Ldot"]/self.profiles["Mdot"]
        r_c = lkep**2. / G / (self.Mbbh*smass_to_g)
        r_g = G*(self.Mbbh*smass_to_g)/c/c
        self.profiles["accretion_efficiency"] = (10*r_g/r_c)
        self.profiles["accretion_efficiency"][self.profiles["accretion_efficiency"]>1] /= self.profiles["accretion_efficiency"][self.profiles["accretion_efficiency"]>1]



    def _interp_profiles_all(self):
        '''
        Creates interpolated profiles for each attribute that has been defined

        interp1d is scipy interp1d
        '''
        # if you haven't interpolated profiles previously, initialize attribute
        if not hasattr(self,"iprofiles"):
            self.iprofiles = {}
        for quantity in self.profiles:
            self.iprofiles[quantity] = interp1d(self.D_arr,self.profiles[quantity])

    def _interp_profile(self,attr,override=0):
        '''
        Creates an interpolated profile for a specific attributes

        Parameters

        attr           the name of the attribute you want to add to interpolated profiles (string)
        override       if == 0, make sure attribute hasn't already been initialized (boolean)
        '''


        # if you haven't interpolated profiles previously, initialize attribute
        if not hasattr(self,'iprofiles'):
            self.iprofiles = {}

        if not override:
            # make sure this attribute is in your AGN profile
            assert (attr in self.profiles)
            # make sure this attribute has not been interpolated yet
            assert (attr not in self.iprofiles)

        self.iprofiles[attr] = interp1d(self.D_arr,self.profiles[attr])

    ## NK: old function
    def _integrate_profile(self,integrand,attr_name,interp=1,override=0):
        """
        Integrand is a function to integrate cumulatively over distance profile,
        attr_name is the name in which to save it
        """
        if not override:
            assert (attr_name not in self.profiles)
            assert (attr_name not in self.iprofiles)

        D = self.D_arr
        integrated = np.array([0])
        for i in range(1,self.N):
            if self.verbose: print("%4.2f %% done integrating..." % (100.*(1.0*i)/self.N))
            integrated = np.concatenate((integrated,np.array([quad(integrand,D[0],D[i])[0]])))

        self.profiles[attr_name] = integrated
        if interp:
            self._interp_profile(attr_name,override=override)

    ## NK: old function
    def adot_at_a_matrix(self,a,with_gw=1,with_drg=1):
        ''' recalculates adot_a profiles given a an array of a's (in au)
            returns a matrix where the rows are a's and the columns are
            distance in the agn '''

        a    = a[:,None]*au_to_cm
        Ldot = self.profiles["Ldot"][None,:]
        Mdot = self.profiles["Mdot"][None,:]
        mach = self.profiles["mach"][None,:]
        Rb   = self.profiles["Rb"][None,:]

        # convenience functions
        machfunc1 = lambda m: (m**3./(8.+4*m**2.) + 1)
        machfunc2 = lambda m: (1. + 0.5*m**2.)


        this_L = np.sqrt(G/16.)*(self.Mbbh*smass_to_g)**(1.5)*(a)**(0.5) # Angular momentum of BBH in cgs
        adot_a_inf = 2.*Ldot/this_L - Mdot/(self.Mbbh*smass_to_g)

        lam = lambda_s(self.g)
        adot_a_drg = -1. * (Mdot/(self.Mbbh*smass_to_g))*(8**0.5)*lam*machfunc2(mach)**(0.5)*machfunc1(mach)**(-1.)*(1. + 4.*a/Rb)**(-0.5)

        adot_a_gw = (-1./4)/( (5./64.) * (c**5./G**3.) * (a**4.) / (smass_to_g*self.Mbbh)**3.)

        adot_a = np.copy(adot_a_inf)
        if with_drg: adot_a += adot_a_drg
        if with_gw:  adot_a += adot_a_gw
        return adot_a

    ## NK: old function
    def adot_at_D_matrix(self,a,with_gw=1,with_drg=1):
        ''' assumes binary inspirals at the orbital timescale
        '''

        ## calculate adot_a first
        a    = a[:,None]*au_to_cm
        Ldot = self.profiles["Ldot"][None,:]
        Mdot = self.profiles["Mdot"][None,:]
        mach = self.profiles["mach"][None,:]
        Rb   = self.profiles["Rb"][None,:]

        # convenience functions
        machfunc1 = lambda m: (m**3./(8.+4*m**2.) + 1)
        machfunc2 = lambda m: (1. + 0.5*m**2.)


        this_L = np.sqrt(G/16.)*(self.Mbbh*smass_to_g)**(1.5)*(a)**(0.5) # Angular momentum of BBH in cgs
        adot_a_inf = 2.*Ldot/this_L - Mdot/(self.Mbbh*smass_to_g)

        lam = lambda_s(self.g)
        adot_a_drg = -1. * (Mdot/(self.Mbbh*smass_to_g))*(8**0.5)*lam*machfunc2(mach)**(0.5)*machfunc1(mach)**(-1.)*(1. + 4.*a/Rb)**(-0.5)

        adot_a_gw = (-1./4)/( (5./64.) * (c**5./G**3.) * (a**4.) / (self.Mbbh*smass_to_g)**3.)

        adot_a = np.copy(adot_a_inf)
        if with_drg: adot_a += adot_a_drg
        if with_gw:  adot_a += adot_a_gw


        # this gives adot(D)
        adot = adot_a * a


        # now, treating our time variables da/dD = da/dt * (dt/dD) = da/dt / (dD/dt / D ) / D
        Ddot_D = self.profiles["Ddot_D"][None,:]
        D      = self.D_arr[None,:] * pc_to_cm

        dadD = adot/D/Ddot_D

        return dadD

    ## NK: old function
    def _integrate_profile_nonlinear(self,f,attr0,D_span,t_eval=None):#,args=None):
        '''
        Do a nonlinear integration of some integrand, where
        d[attr]/dD = f(D, [attr])
        D_span = (D0,DF)
        from attr = attr0 and D = D0 --> DF to some final solution
        f is a function of [attr], D

        Currently just a wrapper function for solve_ivp
        '''
        solution = solve_ivp(f,D_span,attr0,t_eval=t_eval,rtol=1e-6,atol=1e-9)#,args=args)
        return solution.t,solution.y

    def _migrate_and_evolve_sim(self,t_span,D0,M0,a0,fudge=0.5762437942007488,sprinkle=0.29814743866717525,modified=0,photontrapping=0,RadEff=1e-2):
        '''
        Here we evolve a, M, D in disk for an embedded BBH

        Parameters
        t_span                pair of initial and final times to integrate. should be a list of ([0, tmax]) where both are floats
        D0                    initial binary distance in disk (pc)
        M0                    initial binary mass (solar mass)
        a0                    initial binary separation (au)
        fudge                 scaling factor for Mdot
        sprinkle              scaling factor for adot_drag
        modified              if == 0, use pure hydrodynamic rates, if == 1, use Eddington-limited prescriptions (boolean)
        photontrapping        only matters if modified==1. if ==1, use photon trapping prescription for super-Eddington drag. (boolean)
        RadEff                radiative efficiency of accretion flow onto binary (float) (deprecated)
        '''
        # convert to cgs
        D0     *= pc_to_cm
        M0     *= smass_to_g
        a0     *= au_to_cm

        def Ddot(d,M,self):
            '''
            Migration rate due to static torques in Paardekooper 2014

            Parameters
            d          ~ distance in AGN disk (cm)
            M          ~ Mass of binary (grams)
            self       ~ self class
            '''
            q = M/(self.Msmbh*smass_to_g)
            t_migr = (np.pi/2.)*self.iprofiles["H_R"](d/pc_to_cm)**2./self.iprofiles["qd"](d/pc_to_cm)/q/self.iprofiles["omega"](d/pc_to_cm)
            return d*(-0.8-self.iprofiles["torque_alpha"](d/pc_to_cm)-0.9*self.iprofiles["torque_beta"](d/pc_to_cm))/t_migr

        def Mdot(d,M,self,fudge=1, RadEff=1e-2):
            '''
            Mass accretion rate of binary

            Parameters
            d          ~ distance in AGN disk (cm)
            M          ~ Mass of binary (grams)
            fudge      ~ scaling factor for mass accretion rate
            RadEff     ~ radiative efficiency of binary accretion flow
            '''


            # convenience functions
            machfunc1 = lambda m: (m**3./(8.+4*m**2.) + 1)
            machfunc2 = lambda m: (1. + 0.5*m**2.)

            H_R = self.iprofiles["H_R"](d/pc_to_cm)
            rho = self.iprofiles["rho"](d/pc_to_cm)
            cs  = self.iprofiles["cs"](d/pc_to_cm)
            q = M/(self.Msmbh*smass_to_g)
            mach = (0.5*q)**(1./3.) * (H_R)**(-1.)
            Rb   = M*G/cs**2.
            Rbw  = Rb/machfunc2(mach)

            if modified:
                mdot     = fudge * 4*np.pi*Rbw**2.*rho*cs*machfunc1(mach)

                mdot_Edd = binary._Edd_Mdot(M/smass_to_g,RadEff=RadEff)
                eff_edd = mdot_Edd/mdot

                return mdot*min(eff_edd,1)
            else:
                return fudge * 4*np.pi*Rbw**2.*rho*cs*machfunc1(mach)

        def adot(d,M,a,self,fudge=1, sprinkle=1,RadEff=1e-2):
            '''
            Inspiral rate rate of binary

            Parameters
            d          ~ distance in AGN disk (cm)
            M          ~ Mass of binary (grams)
            fudge      ~ scaling factor for mass accretion rate
            sprinle    ~ scaling factor for drag rate
            RadEff     ~ radiative efficiency of binary accretion flow
            '''
            # convenience functions for mach number
            machfunc1 = lambda m: (m**3./(8.+4*m**2.) + 1)
            machfunc2 = lambda m: (1. + 0.5*m**2.)

            # We already have M as a function of d as we migrate inwards
            L    = np.sqrt(G/16.)*M**(1.5)*np.abs(a)**(0.5)
            H_R  = self.iprofiles["H_R"](d/pc_to_cm)
            rho  = self.iprofiles["rho"](d/pc_to_cm)
            cs   = self.iprofiles["cs"](d/pc_to_cm)
            q    = M/(self.Msmbh*smass_to_g)
            mach = (0.5*q)**(1./3.) * (H_R)**(-1.)
            Rb   = M*G/cs**2.
            Rbw  = Rb/machfunc2(mach)

            mdot = fudge * 4*np.pi*Rbw**2.*rho*cs*machfunc1(mach)

            # Ldot is negligible in real systems, so we neglect it
            # Ldot = (np.pi/6.)*rho*cs**2.*Rbw**3. * machfunc2(mach)**(-2.)

            if modified:
                ## eddington
                mdot_Edd = binary._Edd_Mdot(M/smass_to_g,RadEff=RadEff)
                eff_edd = min(mdot_Edd/mdot, 1)


                if photontrapping:
                        ## drag efficiency
                        ## assumes that Mdot ~ (r/r_pt), scaling between mdot at large radii and mdot_Edd at r_pt
                        ## we want the fraction of ejected material at the semimajor axis, Mdot(r=a)/Mdot_ext, which
                        ## we assume the gas density is reduced by, which affects the efficiency of accretion
                        # schwarzchild radius for individual BBH, cgs
                        r_schwarz = G*M/c/c
                        # photon trapping radius
                        r_pt      = r_schwarz/eff_edd
                        if r_pt > a: # if photon trapping radius is greater than sma, all unbound mass is ejected beyond sma
                            eff_drag = np.copy(eff_edd)
                            print("r_pt > a")
                        else:        # if r_pt is less than sma, only some fraction of total ejected mass is lost at the sma
                            eff_drag = min(1, (a/r_pt)*eff_edd)
                            print("r_pt < a")
                else:
                    ## Eddington limited drag efficiency
                    eff_drag = np.copy(eff_edd)
            else:
                eff_edd  = 1
                eff_drag = 1

            # now calculate instantaneous inspiral rates

            if hasattr(self,'g'):
                lam = lambda_s(self.g)
            else:
                lam = lambda_s(1.1)

            # there is a factor of fudge inside mdot that should be cancelled
            adot_a_drg = -1. * (sprinkle/fudge) * (mdot/M)*(8**0.5)*lam*machfunc2(mach)**(0.5)*machfunc1(mach)**(-1.)*(1. + 4.*a/Rb)**(-0.5)

            ## can apply mdot efficiency now as well
            mdot       = mdot*eff_edd

            ## apply drag efficiency
            print("eff_drag = ", eff_drag, "eff_edd = ", eff_edd)
            adot_a_drg = adot_a_drg*eff_drag


            # ignoring Ldot term
            adot_a_inf = -1*mdot/M
            #adot_a_inf = 2.*Ldot/L - mdot/M


            adot_a_gw = (-1.)/( (5./64.) * (c**5./G**3.) * (a**4.) / (M)**3.)

            adot_a = np.copy(adot_a_inf)
            adot_a += adot_a_drg
            adot_a += adot_a_gw
            print("adot_a's:", adot_a_inf, adot_a_drg, adot_a_gw, "a = ", a/au_to_cm)

            return adot_a*a

        ## integrate
        # dy/dt = f(t,y)
        rhs = lambda t, vec: [Ddot(vec[0],vec[1],self),Mdot(vec[0],vec[1],self,fudge=fudge),adot(vec[0],vec[1],vec[2],self,fudge=fudge,sprinkle=sprinkle)]

        solution = solve_ivp(rhs,t_span,[D0,M0,a0],rtol=1e-7,atol=1e-10)

        return solution



class AGN_model(object):
    '''
    This is the base class for AGN models, and contains things that are specific to all AGN models.
    '''
    def __init__(self,Msmbh,D,alpha,EddRatio,RadEff):
        self.Msmbh = Msmbh * smass_to_g   # supermassive black hole mass in grams
        self.D     = D * pc_to_cm         # distance of BBH from SMBH is in cm
        self.alpha = alpha                # (SS only) alpha viscosity parameters
        self.EddRatio = EddRatio          # Eddington Ratio
        self.RadEff   = RadEff            # Radiative efficiency
        self.Mdot  = self.EddRatio * (1.26e38)*(Msmbh) / c**2. / self.RadEff # convert from Eddington ratio to grams/sec

        # Calculate omega
        self.omega = np.sqrt(G*self.Msmbh/self.D**3.)

    def _calc(self):
        '''
        Calculate additional quantities; usually the profile needs to be calculated first.
        '''
        self.H_R = self.H/self.D
        self.Q   = (self.omega**2.)/(2.*np.pi*G*self.rho)

class sirko_goodman(AGN_model):
    '''
    https://arxiv.org/pdf/astro-ph/0209469.pdf
    '''
    def __init__(self,Msmbh,D,alpha,EddRatio,RadEff,b=0,tau_guess=1.0, pgas_guess=1.0):
        super(sirko_goodman,self).__init__(Msmbh,D,alpha,EddRatio,RadEff)

        '''
        Sirko+Goodman 2003 must be numerically solved

        p = b, tau_guess, pgass_guess
        '''

        # b = 1: alpha \propto gas pressure
        # b = 0: alpha \propto total pressure
        self.b     = b

        # adjust Mdot to be Mdot' in SG03
        self.Mdot  = self.Mdot*(1. - np.sqrt(2.0*G*self.Msmbh/c/c/self.D/4./0.1)) # modify Mdot for inner boundary

        # executing calls to produce
        init_guess = np.array([np.log10(tau_guess), pgas_guess])
        sol = least_squares(self.solve_matrix, init_guess, bounds = ([-4,0],[6,1e5]),method='trf',ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=1000)#, method='lm')

        if not sol.success:
            print(sol)
            print("Matrix solver failed! Error = ", self.solve_matrix(sol.x), "D [pc] = ", D, "tau = ", 10**sol.x[0], "pgas = ", sol.x[1])
            self.succeed = 0
        else:
            self.succeed = 1

        tau = 10.**sol.x[0]
        pgas = sol.x[1]

        self.Teff,self.T,self.tau,self.sigma,self.beta,self.cs,self.prad,self.pgas,self.rho,self.H,self.kappa = self.both_sets(tau,pgas)

    def set1_equation(self,tau,pgas):
        Teff  = ((3./8./np.pi)*self.Mdot*(self.omega**2.)/sig_sb)**(1./4.)
        prad  = tau*sig_sb*(Teff**4.)/(2.*c)
        T     = ((3./8.)*tau + 0.5 + 0.25/tau)**(1./4.) * Teff
        rho   = pgas*mH/(k_b*T)
        cs    = ( (pgas + prad)/rho )**0.5
        beta  = pgas/(pgas+prad)
        h     = cs/self.omega
        sigma = 2.*rho*h
        kappa = 10.**opacity_function(rho/(T/1e6)**3.,T)
        if np.shape(kappa) > (): kappa = kappa[0]

        return Teff,T,tau,sigma,beta,cs,prad,pgas,rho,h,kappa


    def set2_equation(self,tau,pgas):
        # A couple of equations we can get right off the bat
        rho   = self.omega**2. / (2. *np.pi*G*1.0)
        T     = pgas*mH/(rho*k_b)
        Teff  = T/((3./8.)*tau + 0.5 + 1./(4.*tau))**(1./4.)
        prad  = tau*sig_sb*(Teff**4.)/(2.*c)
        beta  = pgas/(pgas+prad)
        cs    = ( (pgas + prad)/rho )**0.5
        h     = cs/self.omega
        sigma = 2.*rho*h
        kappa = 10.**opacity_function(rho/(T/1e6)**3.,T)
        if np.shape(kappa) > (): kappa = kappa[0]

        return Teff,T,tau,sigma,beta,cs,prad,pgas,rho,h,kappa

    def both_sets(self,tau,pgas):  # Will call set1_equation, set2_equation
        Teff,T,tau,sigma,beta,cs,prad,pgas,rho,h,kappa = self.set1_equation(tau,pgas)
        Q = (self.omega**2.)/(2.*np.pi*G*rho)
        if np.shape(Q) > (): Q = Q[0]
        if Q < 1.: # Set 1 equations are Toomre unstable, use set 2 equations
            Teff,T,tau,sigma,beta,cs,prad,pgas,rho,h,kappa = self.set2_equation(tau,pgas)

        return Teff,T,tau,sigma,beta,cs,prad,pgas,rho,h,kappa

    def solve_matrix(self,guess): # Will call set1_equation, set2_equation
        tau  = 10.**guess[0]
        pgas = guess[1]
        Teff,T,tau,sigma,beta,cs,prad,pgas,rho,h,kappa = self.both_sets(tau,pgas)


        return np.array([T**4 - ((3./8.)*tau + 0.5 + 0.25/tau)*Teff**4.,
                             tau - kappa*sigma/2.0,
                             beta**(self.b)*(cs**2.)*sigma - self.Mdot*self.omega/(3.*np.pi*self.alpha),
                             prad - tau*sig_sb*(Teff**4.)/(2.*c),
                             pgas - rho*k_b*T/mH,
                             beta - pgas/(pgas+prad),
                             sigma - 2.*rho*h,
                             h - cs/self.omega,
                             cs**2. - (pgas+prad)/rho,
                            ])

    def p(self):
        return np.array([self.b,self.tau,self.pgas])


## build opacity table
opacity_function = init_opacity()
