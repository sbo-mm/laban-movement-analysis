import requests
import warnings
from html.parser import HTMLParser

class CMUHTMLSubjectParser(HTMLParser):
    
    def __init__(self):
        super().__init__()
        self.reset_all()
        
        self.subject_number = 1
        self.data_idx       = 0
        self.data_ignore    = [
            '\'', '\n\n', 'tvd', 'c3d', 
            'mpg', 'Animated', 'Feedback'
        ]
        
    def reset_all(self):
        self.reset()
        self.master = {}
        self.tr_enabled   = False
        self.data_enabled = False
        self.data_zero    = None
        self.cur_url      = None
        self.data         = ['' for _ in range(4)]
        self.list_comp    = self.data[:]        
        self.data_idx     = 0
    
    def set_subject(self, subject_num):
        self.subject_number = subject_num
        self.master["Subject"] = subject_num
    
    def handle_starttag(self, tag, attrs):
        if tag == 'tr':        
            self.tr_enabled = True
        
        if not self.tr_enabled:
            return
        
        if tag == 'a':
            if len(attrs) == 1:
                url = attrs[0][1]
                toks = url.split('.')
                if toks[-1] == 'asf':
                    self.master['asf'] = url.encode('utf-8')
            
            self.cur_url = attrs[0][1]
              
    def handle_data(self, data):
        if self.tr_enabled:
            self.data_zero = data
            if isinstance(self.data_zero, str):
                if self.data_zero.isnumeric():
                    self.data_enabled = True
        
        if not self.data_enabled:
            return 
        
        if data in self.data_ignore:
            return
        
        if self.data_idx == 2:
            self.data[self.data_idx] = self.cur_url
        else:
            self.data[self.data_idx] = data
        
        self.data_idx += 1
        
    def handle_endtag(self, tag):
        if tag == 'tr':
            self.tr_enabled = False
            if self.data != self.list_comp:                
                d = {
                    'Category': None,
                    'Description': self.data[1],
                    'amc': self.data[2].encode('utf-8'),
                    'FrameRate': int(self.data[3])
                }
                
                trial = 'Trial{0}'.format(self.data[0])
                self.master[trial] = d
                
                self.data = self.list_comp[:]
                self.data_enbaled = False
                self.data_idx = 0
                

class CMUScanner(object):
    
    def __init__(self):
        self.html_parser = CMUHTMLSubjectParser()
        self.base_url = 'http://mocap.cs.cmu.edu/search.php?subjectnumber={0}'
        self.allowed_subjects = [snum for snum in range(1, 145)]
    
    def __get_url_content(self, num):
        url = self.base_url.format(num)
        resp = requests.get(url, allow_redirects=True)
        
        cont_t = resp.headers.get('content-type')
        assert cont_t == 'text/html', "Invalid content {0}".format(cont)
        
        return str(resp.content)
    
    def __parse_html(self, html):
        self.html_parser.feed(html)
        return dict(self.html_parser.master)
        
    def scan_cmu(self, subject_number=1):
        if subject_number not in self.allowed_subjects:
            warnings.warn(
                "Subject number not allowed, using default (1)"
            )
            subject_number = 1
        
        self.html_parser.reset_all()
        
        self.html_parser.set_subject(subject_number)
        cont = self.__get_url_content(subject_number)
        return self.__parse_html(cont)
