import sys; sys.path.insert(1, './cython/')
from asfamcparser import CyReadASFAMC
from exceptions import ResourceExhaustedError, ResourceNotLoadedError

class AcclaimFileHandler(CyReadASFAMC):
    
    def __init__(self, asf_path=None, amc_path=None, TYPE=None):
        super().__init__()
        self.asf_loaded      = False
        self.amc_loaded      = False
        self.resources_freed = False
        
        self.last_asf_path   = b""
        self.last_amc_path   = b""
        
        self.motion_data = []
        self.hierarchy   = {}
        
        if asf_path and amc_path and TYPE:
            self.manual_parse_asf(asf_path, TYPE)
            self.manual_parse_amc(amc_path, TYPE)
            self.motion_data = self.py_get_poses()
            self.hierarchy   = self.py_get_hierarchy()
       
    def manual_parse_asf(self, asf_path, TYPE):
        if self.resources_freed:
            raise ResourceExhaustedError(
                "All resources has been freed"
            )        
        
        if asf_path == self.last_asf_path:
            return
        
        ret = self.py_parse_asf(asf_path, TYPE)
        if ret < 0:
            raise FileNotFoundError(
                2, "No such file or directory: '%s'" % asf_path
            )
        
        self.last_asf_path = asf_path
        self.asf_loaded = True
    
    def manual_parse_amc(self, amc_path, TYPE):
        if self.resources_freed:
            raise ResourceExhaustedError(
                "All resources has been freed"
            )
        
        if amc_path == self.last_amc_path:
            return        
        
        ret = self.py_parse_amc(amc_path, TYPE)
        if ret < 0:
            raise FileNotFoundError(
                2, "No such file or directory: '%s'" % amc_path
            )
        
        self.last_amc_path = amc_path
        self.amc_loaded = True
    
    def load_poses(self):
        if not (self.asf_loaded and self.amc_loaded):
            raise ResourceNotLoadedError(
                "Motion Files have not been loaded"
            )
        
        if self.resources_freed:
            raise ResourceExhaustedError(
                "All resources has been freed"
            )
            
        self.motion_data = self.py_get_poses()
        self.hierarchy   = self.py_get_hierarchy()
        return (self.motion_data, self.hierarchy)
        
    def manual_dealloc(self):
        if self.resources_freed:
            raise ResourceExhaustedError(
                "All resources has been freed"
            )
        
        self.dealloc_all()
        self.resources_freed = True
        
    def __del__(self):
        if not self.resources_freed:
            self.manual_dealloc()
