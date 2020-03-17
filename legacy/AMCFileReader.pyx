# coding: utf-8
# cython: boundscheck=False

from libc.stdlib cimport malloc, realloc, free, atof, atoi 
from libc.stdio cimport fopen, fclose, FILE, fseek, SEEK_END, SEEK_SET
from libc.stdio cimport ftell, fread, getline
from libc.string cimport strcmp, strlen, memcpy, strcpy, strtok, strcspn

from cpython.pycapsule cimport *

import numpy as np
cimport numpy as np

cdef extern from "math.h":
    float sin(float x)
    float cos(float x)
    int abs(int x)
    
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t 


cdef np.ndarray[DTYPE_t, ndim=2] rotXYZ(float x, float y, float z):
    cdef np.ndarray[DTYPE_t, ndim=2] out
    cdef float c1, c2, c3, s1, s2, s3
    c1 = cos(x); c2 = cos(y); c3 = cos(z)
    s1 = sin(x); s2 = sin(y); s3 = sin(z)
    
    out = np.array([
        [c2*c3         , -c2*s3        , s2    ],
        [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
        [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2 ]
    ], dtype=DTYPE)
    
    return out


cdef float PI = <DTYPE_t>np.pi


cdef float deg2rad(float angle):
    return (PI / 180.) * angle


cdef np.ndarray[DTYPE_t, ndim=2] euler2mat(float *angles):
    cdef np.ndarray[DTYPE_t, ndim=2] res = rotXYZ(
        deg2rad(angles[0]),
        deg2rad(angles[0]),
        deg2rad(angles[0])
    )
    
    return res 


cdef np.ndarray[DTYPE_t, ndim=2] matmul(float [:, :] m1, float [:, :] m2):
    cdef np.ndarray[DTYPE_t, ndim=2] res 
    cdef int i, j, k
    cdef DTYPE_t s
    
    res = np.zeros((m1.shape[0], m2.shape[1]), dtype=DTYPE)
    
    for i in range(m1.shape[0]):
        for j in range(m2.shape[1]):
            s = 0.
            for k in range(m1.shape[1]):
                s += m1[i, k] * m2[k, j]

            res[i, j] = s
    
    return res
    

cdef int CERROR = -9


cdef class CyFile:
    
    cdef:
        FILE *fp
        char *filename
        bint is_open
        
    def __init__(self):
        self.is_open = 0
        self.filename = <char *>malloc(100 * sizeof(char))
        
    cdef int open_file(self, char *filename):
        self.fp = fopen(filename, "r")
        if self.fp == NULL:
            print("Error opening file")
            return CERROR
        
        strcpy(self.filename, filename)
        self.is_open = 1
        return 0
    
    cdef int close_file(self):
        cdef int ret = 0
        if self.is_open:
            ret = fclose(self.fp)
            self.is_open = 0

            if ret != 0:
                print("Error closing file")
                return CERROR
        
        return 0
        
    def __CyDealloc__(self):
        if self.filename:
            free(self.filename)
        
        if self.is_open:
            self.close_file()
        

cdef class CyTokenParser(CyFile):

    cdef int MAXTOKS    
    cdef ssize_t line_size
    cdef char *tmpstr
    cdef size_t line_buf_size

    def __init__(self):
        super().__init__()
        self.MAXTOKS = 5
        self.line_size = 0
        self.line_buf_size = 0
        
        self.tmpstr = <char *>malloc(sizeof(char))
        
    cdef int strip_line(self, char *line):
        if line == NULL:
            return CERROR

        line[strcspn(line, "\r\n")] = 0
        return 0

    cdef int __str_dup(self, char *instr, ssize_t instrlen):
        cdef char *mem = <char *>realloc(
            self.tmpstr, (instrlen + 1) * sizeof(char)
        )           
        
        if mem:
            self.tmpstr = mem
        else:
            return CERROR
        
        strcpy(self.tmpstr, instr)
        return 0

    cdef int split_line(self, char *line, ssize_t line_size, char **line_tokens):
        cdef:
            char *token
            int ret_idx
        
        if self.__str_dup(line, line_size) < 0:
            return CERROR
        
        token = strtok(self.tmpstr, " ")
        ret_idx = 0

        while token != NULL:
            strcpy(line_tokens[ret_idx], token)

            ret_idx += 1
            if ret_idx == self.MAXTOKS:
                break

            token = strtok(NULL, " ")            

        return ret_idx

    cdef ssize_t read_line(self, char **line_buf):
        cdef ssize_t line_size

        if not self.is_open:
            return CERROR

        if line_buf == NULL:
            return CERROR

        line_size = getline(line_buf, &self.line_buf_size, self.fp)
        return line_size
    
    cdef int read_line_tokens(self, char *line_buf, char **line_tokens):
        cdef:
            ssize_t line_size
            int ret_idx
        
        line_size = self.read_line(&line_buf)
        
        if line_size < 0:
            return CERROR
        
        self.strip_line(line_buf)
        ret_idx = self.split_line(line_buf, line_size, line_tokens)
        
        if ret_idx < 0:
            return CERROR
        
        return ret_idx
        
    def __CyDealloc__(self):
        super().__CyDealloc__()
        
        if self.tmpstr:
            free(self.tmpstr)
        

cdef class CyAcclaimParser(CyTokenParser):
    
    cdef:
        char *line_buf
                
        size_t MAXTOK_SIZE
        char **line_tokens
                
    def __init__(self):
        super().__init__()
        self.line_buf = NULL        

        self.MAXTOK_SIZE = 100
        self.line_tokens = <char **>malloc(
            self.MAXTOKS * sizeof(char *)
        )
        
        cdef int i
        for i in range(self.MAXTOKS):
            self.line_tokens[i] = <char*>malloc(
                self.MAXTOK_SIZE * sizeof(char)
            )        
        
    def __CyDealloc__(self):
        super().__CyDealloc__()    
        
        if self.line_buf:
            free(self.line_buf)
        
        cdef int i
        for i in range(self.MAXTOKS):
            if self.line_tokens[i]:
                free(self.line_tokens[i])
                
        if self.line_tokens:
            free(self.line_tokens)

            
cdef struct ASFRoot:
    char order[6][3]
    char axis[4]
    float positions[3]
    float orientations[3]

cdef struct ASFParams:
    unsigned int joint_id
    char name[15]
    float direction[3][1]
    float length
    float axis[3]
    char dof[3][3]
    float limits[6]                
    
    ASFParams *parent
    ASFParams **children
    int num_children
    
    float C[3][3]
    float Cinv[3][3]


cdef float NOLIMIT = -1000.


cdef class CyASFParser(CyAcclaimParser):
     
    cdef:
        ASFRoot *asf_root
        ASFParams **asf_params
        int MAXPARAMS
        
        int lims_size
        char **lims_buffer
        
        ASFParams *root
        dict asf_ptrs
        int num_params

    def __init__(self):
        super().__init__()
        self.asf_root = <ASFRoot *>malloc(sizeof(ASFRoot))
        
        self.MAXPARAMS = 50
        self.asf_params = <ASFParams **>malloc(
            self.MAXPARAMS * sizeof(ASFParams *)
        )
        
        cdef int i, j, k
        for i in range(self.MAXPARAMS):
            self.asf_params[i] = <ASFParams *>malloc(
                sizeof(ASFParams)
            )
            
            self.asf_params[i].parent = NULL
            self.asf_params[i].children = NULL
            self.asf_params[i].num_children = 0
            
            for j in range(3):
                strcpy(self.asf_params[i].dof[j], b"NA\0")
                
            for k in range(6):
                self.asf_params[i].limits[k] = NOLIMIT
        
        self.lims_size = self.MAXTOKS * sizeof(char *)
        self.lims_buffer = <char **>malloc(
            self.lims_size
        )
        
        
    cdef int parse_asf_root_strict(self):
        cdef int i = 0
        cdef int ret_idx = 0
        
        while ret_idx >= 0:            
            if strcmp(self.line_tokens[0], b":root") == 0:
                break
                
            ret_idx = self.read_line_tokens(self.line_buf, self.line_tokens)

        while ret_idx >= 0:
            if strcmp(self.line_tokens[0], b":bonedata") == 0:  
                break
                
            if strcmp(self.line_tokens[0], b"order") == 0:
                for i in range(1, ret_idx):
                    strcpy(self.asf_root.order[i-1], self.line_tokens[i])
                
            elif strcmp(self.line_tokens[0], b"axis") == 0:
                strcpy(self.asf_root.axis, self.line_tokens[1])
                
            elif strcmp(self.line_tokens[0], b"position") == 0:
                for i in range(1, ret_idx):
                    self.asf_root.positions[i-1] = atof(self.line_tokens[i])
                
            elif strcmp(self.line_tokens[0], b"orientation") == 0:
                for i in range(1, ret_idx):
                    self.asf_root.orientations[i-1] = atof(self.line_tokens[i])
            
            ret_idx = self.read_line_tokens(self.line_buf, self.line_tokens)
            
        return ret_idx
    
    
    cdef int parse_limits(self, char **limits, int idx_counter, int i):
        cdef int k = 0
        cdef int idx = 0
        
        if limits[0][0] == b'(':
            strcpy(limits[0], limits[0] + 1)        
        
        cdef size_t slen = strlen(limits[0])
        if limits[0][slen - 1] == b')':
            limits[0][slen - 1] = 0
        
        for k in range(2):
            idx = i + k
            self.asf_params[idx_counter].limits[idx] = atof(limits[k])
            
        return 0
    
    cdef int parse_params(self, int init_idx, int idx_counter):
        cdef int i = 0
        cdef int k = 0
        cdef int ndof = 0
        cdef int ret_idx = init_idx
        
        while ret_idx >= 0:
            
            if strcmp(self.line_tokens[0], b"end") == 0:
                break
            
            if strcmp(self.line_tokens[0], b"id") == 0:
                self.asf_params[idx_counter].joint_id = atoi(self.line_tokens[1])

            elif strcmp(self.line_tokens[0], b"name") == 0:
                strcpy(self.asf_params[idx_counter].name, self.line_tokens[1])

            elif strcmp(self.line_tokens[0], b"direction") == 0:
                for i in range(1, ret_idx):
                    self.asf_params[idx_counter].direction[i-1][0] = atof(self.line_tokens[i])  

            elif strcmp(self.line_tokens[0], b"length") == 0:
                self.asf_params[idx_counter].length = atof(self.line_tokens[1])

            elif strcmp(self.line_tokens[0], b"axis") == 0:
                for i in range(1, ret_idx - 1):
                    self.asf_params[idx_counter].axis[i-1] = atof(self.line_tokens[i])

            elif strcmp(self.line_tokens[0], b"dof") == 0:
                ndof = ret_idx - 1
                for i in range(1, ret_idx):
                    strcpy(self.asf_params[idx_counter].dof[i-1], self.line_tokens[i])

            elif strcmp(self.line_tokens[0], b"limits") == 0:
                memcpy(self.lims_buffer, self.line_tokens + 1, self.lims_size)
                for i in range(ndof):
                    if strcmp(self.asf_params[idx_counter].dof[i], b"rx") == 0:
                        k = 0
                    elif strcmp(self.asf_params[idx_counter].dof[i], b"ry") == 0:
                        k = 1
                    else:
                        k = 2
                    
                    self.parse_limits(self.lims_buffer, idx_counter, k * 2)
                    
                    if i == ndof - 1:
                        break
                        
                    self.read_line_tokens(self.line_buf, self.line_tokens)
                    memcpy(self.lims_buffer, self.line_tokens , self.lims_size)
                    
            ret_idx = self.read_line_tokens(self.line_buf, self.line_tokens)
        
        return 0
    
    cdef int parse_asf_params_strict(self):
        cdef int idx_counter = 0
        cdef int init_idx = 0
        
        while self.read_line_tokens(self.line_buf, self.line_tokens) >= 0:            
            
            if strcmp(self.line_tokens[0], b":hierarchy") == 0:
                break
            
            if idx_counter == self.MAXPARAMS:
                break
                
            if strcmp(self.line_tokens[0], b"begin") == 0:
                init_idx = self.read_line_tokens(self.line_buf, self.line_tokens)
                self.parse_params(init_idx, idx_counter)
            
            idx_counter += 1


        if idx_counter < self.MAXPARAMS:
            self.asf_params = <ASFParams **>realloc(
                self.asf_params, idx_counter * sizeof(ASFParams *)
            )

        self.num_params = idx_counter
        return 0
    
    
    cdef ASFParams *create_root(self):
        self.root = <ASFParams *>malloc(sizeof(ASFParams))
        
        self.root.joint_id = 0
        strcpy(self.root.name, b"root")
        self.root.direction[0][0] = 0.
        self.root.direction[1][0] = 0.
        self.root.direction[2][0] = 0.
        self.root.length = 0.
        self.root.axis = [0., 0., 0.]
        strcpy(self.root.dof[0], b"NA\0")
        strcpy(self.root.dof[1], b"NA\0")
        strcpy(self.root.dof[2], b"NA\0")
        self.root.limits = [
            NOLIMIT, NOLIMIT, NOLIMIT,
            NOLIMIT, NOLIMIT, NOLIMIT
        ]
        self.root.parent = NULL
        self.root.children = NULL
        return self.root
    
    cdef dict create_ptr_capsules(self):
        cdef dict ptr_capsules = {}
        
        cdef ASFParams *root = self.create_root()        
        ptr_capsules[root.name] = PyCapsule_New(<void *>root, root.name, NULL)

        cdef int i
        for i in range(self.MAXPARAMS):
            ptr_capsules[self.asf_params[i].name] = PyCapsule_New(
                <void *>self.asf_params[i], self.asf_params[i].name, NULL
            )
            
        return ptr_capsules
    
    cdef ASFParams *release_capsule(self, capsule, char *key):
        cdef ASFParams *param
        
        if not PyCapsule_IsValid(capsule, key):
            return NULL
        
        param = <ASFParams *>PyCapsule_GetPointer(capsule, key)
        return param
    
    cdef int parse_asf_hierarchy(self, int ret_idx, dict ptrs):
        cdef int i
        cdef ASFParams *parent
        cdef ASFParams *child
        
        capsule = ptrs[self.line_tokens[0]]
        parent = self.release_capsule(capsule, self.line_tokens[0])
        
        parent.num_children = ret_idx - 1
        parent.children = <ASFParams **>malloc(
            (ret_idx - 1) * sizeof(ASFParams *)
        )
        
        for i in range(1, ret_idx):
            capsule = ptrs[self.line_tokens[i]]
            child = self.release_capsule(capsule, self.line_tokens[i])
            parent.children[i - 1] = child

            child.parent = <ASFParams *>malloc(sizeof(ASFParams *))
            child.parent = parent
            
        return 0
    
    cdef int parse_asf_hierarchy_strict(self):        
        cdef int ret_idx = 0
        cdef dict ptrs = self.create_ptr_capsules()
        
        while ret_idx >= 0:
                        
            if strcmp(self.line_tokens[0], b":hierarchy") == 0:
                self.read_line_tokens(self.line_buf, self.line_tokens)
                assert strcmp(self.line_tokens[0], b"begin") == 0
                
                ret_idx = self.read_line_tokens(self.line_buf, self.line_tokens)
                break
        
            ret_idx = self.read_line_tokens(self.line_buf, self.line_tokens)
        
        while ret_idx >= 0:
            if strcmp(self.line_tokens[0], b"end") == 0:
                break
            
            self.parse_asf_hierarchy(ret_idx, ptrs)
            
            ret_idx = self.read_line_tokens(self.line_buf, self.line_tokens)
        
        self.asf_ptrs = ptrs
        return 0
    

    cdef int parse_asf(self, char *name):
        if self.open_file(name) == CERROR:
            return CERROR

        self.parse_asf_root_strict()        
        self.parse_asf_params_strict()
        self.parse_asf_hierarchy_strict()

        if self.close_file() == CERROR:
            return CERROR

        return 0
        
    def __CyDealloc__(self):
        super().__CyDealloc__()    
        
        if self.lims_buffer:
            free(self.lims_buffer)
        
        if self.asf_root:
            free(self.asf_root)
                        
        cdef int i
        for i in range(self.MAXPARAMS):
            if self.asf_params[i]:
                if self.asf_params[i].children:
                    free(self.asf_params[i].children)
                
                free(self.asf_params[i])
                
        if self.asf_params:
            free(self.asf_params)
        
            
cdef struct AMCParams:
    char name[15]
    int dsize 
    float *data
            

cdef class CyAMCParser(CyAcclaimParser):
    
    cdef:
        int MAXFRAMES
        AMCParams ***amc_params
        
        int num_frames
        int num_params
        
    def __init__(self):
        super().__init__()
        self.MAXFRAMES = 25000
        
        self.amc_params = <AMCParams ***>malloc(
            self.MAXFRAMES * sizeof(AMCParams **)
        )
        
    cdef int count_params(self):
        cdef int ret_idx = 0
        cdef int c = 0
        while ret_idx >= 0:
            
            if ret_idx == 1:
                break
            
            c += 1
            ret_idx = self.read_line_tokens(self.line_buf, self.line_tokens)
        
        return c
    
    cdef int init_amc(self):
        cdef int ret_idx = 0
        cdef int c = 0
        cdef long int file_pos        
        
        while ret_idx >= 0:
            
            if self.line_tokens[0][0] ==  b'#':
                ret_idx = self.read_line_tokens(self.line_buf, self.line_tokens)
                continue
                
            if self.line_tokens[0][0] == b':':
                ret_idx = self.read_line_tokens(self.line_buf, self.line_tokens)
                continue
            
            if ret_idx == 1:
                file_pos = ftell(self.fp)
                c = self.count_params()
                fseek(self.fp, file_pos, SEEK_SET)
                break
            
            ret_idx = self.read_line_tokens(self.line_buf, self.line_tokens)
            
        return c
    
    cdef int parse_amc_strict(self):
        cdef int i, j, k, K
        cdef int c = 0
        cdef int ret_idx = 0
        cdef bint EOF = 0
        
        cdef AMCParams *amc_ptr
        
        c = self.init_amc()
        
        for i in range(self.MAXFRAMES):
            K = 1
            self.amc_params[i] = <AMCParams **>malloc((c - 1) * sizeof(AMCParams *))
            ret_idx = self.read_line_tokens(self.line_buf, self.line_tokens)
            for j in range(c - 1):
                
                if ret_idx < 0:
                    EOF = 1
                    break
                
                self.amc_params[i][j] = <AMCParams *>malloc(sizeof(AMCParams))
                amc_ptr = self.amc_params[i][j]
                strcpy(amc_ptr.name, self.line_tokens[0])
                
                amc_ptr.dsize = ret_idx - 1
                amc_ptr.data = <float *>malloc(amc_ptr.dsize * sizeof(float))
                
                for k in range(1, ret_idx):
                    amc_ptr.data[k - 1] = atof(self.line_tokens[k])
                
                ret_idx = self.read_line_tokens(self.line_buf, self.line_tokens)
                K += 1
            
            if EOF:
                break
            
            
        if (K % c) != 0:
            self.amc_params = <AMCParams ***>realloc(
                self.amc_params, i * sizeof(AMCParams **)
            )
        
        self.num_frames = i
        self.num_params = c - 1        
        return 0
    
    cdef int parse_amc(self, char *f):
        if self.open_file(f) == CERROR:
            return CERROR

        self.parse_amc_strict()

        if self.close_file() == CERROR:
            return CERROR

        return 0
    
    def __CyDealloc__(self):
        super().__CyDealloc__()
        
        cdef int i, j, k
        for i in range(self.num_frames):
            for j in range(self.num_params):
                
                if self.amc_params[i][j].data:
                    free(self.amc_params[i][j].data)
            
                if self.amc_params[i][j]:
                    free(self.amc_params[i][j])
                
            if self.amc_params[i]:
                free(self.amc_params[i])
        
        if self.amc_params:
            free(self.amc_params)
              
        
cdef class CyAcclaimHandler:
    
    cdef:
        CyASFParser asf_parser
        CyAMCParser amc_parser
        
        char *asf_file_path
        char *amc_file_path
    
    def __init__(self, char *asf_file_path, char *amc_file_path):
        self.asf_parser = CyASFParser()
        self.amc_parser = CyAMCParser()
        self.asf_file_path = asf_file_path
        self.amc_file_path = amc_file_path

    cdef int parse_asf_amc(self):
        if self.asf_parser.parse_asf(self.asf_file_path) == CERROR:
            return CERROR

        if self.amc_parser.parse_amc(self.amc_file_path) == CERROR:
            return CERROR

        return 0

    cdef ASFParams *get_root(self):
        cdef dict ptrs = self.asf_parser.asf_ptrs
        capsule = ptrs[b"root"]
        return self.asf_parser.release_capsule(
            capsule, b"root"
        )
        
    cdef int view_to_buf(self, float[:, :] view, float [3][3] buf):
        cdef Py_ssize_t i, j    
        for i in range(view.shape[0]):
            for j in range(view.shape[1]):
                buf[i][j] = view[i, j]
        
        return 0
        
    cdef int set_global_transforms(self):
        cdef dict ptrs
        cdef ASFParams *param
        cdef np.ndarray[DTYPE_t, ndim=2] C
        cdef np.ndarray[DTYPE_t, ndim=2] Cinv
        
        ptrs = self.asf_parser.asf_ptrs
        
        for key in ptrs.keys():
            capsule = ptrs[key]
            param = self.asf_parser.release_capsule(
                capsule, key
            )
            
            C = euler2mat(param.axis)
            Cinv = np.linalg.inv(C)
            
            self.view_to_buf(C, param.C)
            self.view_to_buf(Cinv, param.Cinv)
            
        return 0
        
            
    cdef AMCParams *search_frame(self, AMCParams **frame, char *key):
        cdef int i
        for i in range(self.amc_parser.num_params):
            if strcmp(frame[i].name, key) == 0:
                return frame[i]
            
        return NULL
    
    
    cdef bint check_lims(self, int offset, float lims[6]):
        cdef int i
        cdef int eq_count = 0
        for i in range(2):
            if lims[offset + i] == NOLIMIT:
                eq_count += 1
        
        return 1 if eq_count == 2 else 0
    
    cdef int set_motion_child(self, AMCParams **frame, ASFParams *child, float parent_pos[3], float [:, :] parent_M, float [:, :] v):
        cdef float pos[3]
        cdef float rot[3]
        cdef float [:, :] C
        cdef float [:, :] M
        cdef float [:, :] P
        
        cdef AMCParams *child_data = self.search_frame(
            frame, child.name
        )         
    
        cdef int idx = 0
        cdef int i, offset
        
        for i in range(3):
            rot[i] = 0.
        
        for i in range(3):
            offset = i * 2
            if not self.check_lims(offset, child.limits):
                rot[i] = child_data.data[idx]
                idx += 1

        C = euler2mat(rot)
        M = matmul(matmul(matmul(parent_M, child.C), C), child.Cinv)
        P = matmul(M, child.direction)
        
        for i in range(3):
            pos[i] = parent_pos[i] + child.length * P[i, 0]
            v[child.joint_id, i] = pos[i]
            
        for i in range(child.num_children):
            self.set_motion_child(frame, child.children[i], pos, M, v)
        
        return 0
    
    cdef int set_motion_root(self, ASFParams *root, AMCParams **frame, float [:, :] v):
        cdef float pos[3]
        cdef float rot[3]
        cdef float [:, :] C
        cdef float [:, :] M
        
        cdef AMCParams *root_data = self.search_frame(
            frame, b"root"
        ) 
        
        cdef int i
        for i in range(3):
            pos[i] = root_data.data[i]
            rot[i] = root_data.data[i + 3]
            v[0, i] = pos[i]
        
        C = euler2mat(rot)
        M = matmul(matmul(root.C, C), root.Cinv)
        
        for i in range(root.num_children):
            self.set_motion_child(frame, root.children[i], pos, M, v)
        
        return 0
    
    
    cdef np.ndarray[DTYPE_t, ndim=2] set_motion(self):
        cdef np.ndarray[DTYPE_t, ndim=3] out
        cdef ASFParams *root 
        cdef AMCParams ***frames 
        cdef int i, num_frames, num_params

        root = self.get_root()
        frames = self.amc_parser.amc_params
        num_frames = self.amc_parser.num_frames
        num_params = self.asf_parser.num_params + 1
        
        out = np.zeros((num_frames, num_params, 3), dtype=DTYPE)
        
        for i in range(num_frames):
            self.set_motion_root(root, frames[i], out[i])
            
        return out
    
    cdef list get_marker_ids(self):
        cdef list li
        cdef dict ptrs 
        cdef ASFParams *param
        
        li = [b'' for _ in range(self.amc_parser.num_params + 2)]
        ptrs = self.asf_parser.asf_ptrs
        for key in ptrs.keys():
            capsule = ptrs[key]
            param = self.asf_parser.release_capsule(
                capsule, key
            )
            
            li[param.joint_id] = param.name
            
        return li
    
    cdef dict get_hierarchy(self):
        cdef dict hier, stub, ptrs
        cdef ASFParams *param
        cdef list children
        cdef int i
        
        hier = {}
        ptrs = self.asf_parser.asf_ptrs
        for key in ptrs.keys():
            capsule = ptrs[key]
            param = self.asf_parser.release_capsule(
                capsule, key
            )
            
            stub = {
                b"name": param.name,
                b"parent": None,
                b"children": None
            }
            
            if param.parent != NULL:
                stub[b"parent"] = param.parent.joint_id
            
            children = []
            for i in range(param.num_children):
                children.append(param.children[i].joint_id)
                
            if children:
                stub[b"children"] = children
                
            hier[param.joint_id] = stub
        
        return hier
        
    
    cdef np.ndarray[DTYPE_t, ndim=3] get_poses(self):
        cdef np.ndarray[DTYPE_t, ndim=3] out
#       if self.parse_asf_amc() == CERROR:
#            return []

        self.set_global_transforms()
        out = self.set_motion()
        return out


    def __CyDealloc__(self):
        self.asf_parser.__CyDealloc__()
        self.amc_parser.__CyDealloc__()
    
            
cdef class CyReadASFAMC(CyAcclaimHandler):

    def __init__(self, asf, amc):
        super().__init__(asf, amc)
        g = self.parse_asf_amc()
        print(g)
#        if self.parse_asf_amc() == CERROR:
#            self.dealloc()
#            # raise error
#            pass

    def py_get_poses(self):
        return self.get_poses()

    def py_get_hierarchy(self):
        return self.get_hierarchy()

    def py_get_marker_ids(self):
        return self.get_marker_ids()

    def dealloc_all(self):
        self.__CyDealloc__()


class PyReadASFAMC(CyReadASFAMC):

    def __init__(self, asf, amc):
        super().__init__(asf, amc)
