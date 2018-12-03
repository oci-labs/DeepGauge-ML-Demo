from datetime import datetime

string = '2018-11-28T16:27:43.627213+00:00'
# print(string)

date_time_obj = datetime.strptime(string, '%Y-%m-%dT%H:%M:%S.%f%z')
print( date_time_obj.strftime('We are the %d, %B %Y') )
