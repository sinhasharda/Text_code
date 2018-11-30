# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:08:30 2016

@author: Sharda.sinha
"""

#!/usr/bin/env python
import web
import sys

sys.path.insert(0,'/root/python-apps/Top3ques')
import top3_hclust_v1

urls = (
    '/top3ques'
)

app = web.application(urls, globals())

class top3ques:             
    def GET(self):
	user_data = web.input()
	companyname=user_data.companyname
	print('------'+companyname)
	print(companyname)
        return User_ip_response.response(companyname)


if __name__ == "__main__":
    app.run()
