from apscheduler.schedulers.blocking import BlockingScheduler

from main import main

sched = BlockingScheduler()


@sched.scheduled_job('interval', hours=1)
def timed_job():
    main()


sched.start()
