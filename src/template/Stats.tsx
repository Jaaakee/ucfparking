/* eslint-disable no-restricted-syntax */
/* eslint-disable guard-for-in */
import { useSelector } from 'react-redux'
import { useDispatch } from 'react-redux'
import { bindActionCreators } from 'redux'
import { actionCreators } from '../redux'
import { StatsCard } from '../stats/StatsCard'
import {  useEffect, useState } from 'react'

const Stats = () => {
const dispatch = useDispatch()
  const {
    // getTodayData
    HandleLastRowData,
    getLastDayData,
    getBarChartData,
    getPieChartData
  } = bindActionCreators(actionCreators, dispatch);
  
  useEffect(() => {
    // dispatch(getTodayData())
    HandleLastRowData()
    getLastDayData()
    getBarChartData()
    getPieChartData()
  }, [])

  // eslint-disable-next-line @typescript-eslint/no-shadow
  const state = useSelector((state: any) => state)
  const [updatedtime, setUpdatedTime] = useState('00:02:01');  

  const { parking, chart } = state
  const { barChartData } = chart
  // eslint-disable-next-line @typescript-eslint/naming-convention
  const { total_data_rows } = parking
  
  //Available Spaces
  let spaces: number = 0
  for (const space in barChartData) {
    spaces += barChartData[space].spaces
  }
 
  // Next Prediction Update
  const nextpredictionUpdate = (utcTime: string | number | Date) => {
  const utcObj = new Date(utcTime);
  // console.log(utcObj);
  const hours = utcObj.getUTCHours();
  const minutes = utcObj.getUTCMinutes();

  let nextYear = utcObj.getUTCFullYear();
  let nextMonth = utcObj.getUTCMonth();
  let nextDate = utcObj.getUTCDate();
  let nextHour = utcObj.getUTCHours();
  const nextMinute = 1;
  const nextSecond = 0;
  
    if ( (hours === 0 && minutes < 1) || hours > 18 || (hours === 18 && minutes >= 1) ) {
      nextHour = 0;
  
      /**
       * Add 6 hours in current utc time.
       * And get utc date components and set hour, minutes and seconds manually.
       *
       */
      const utcTimePlus6HoursInMilliSeconds = utcObj.getTime() + (6*60*60*1000);
      const newUtcObj = new Date(utcTimePlus6HoursInMilliSeconds);
  
      nextYear = newUtcObj.getUTCFullYear();
      nextMonth = newUtcObj.getUTCMonth();
      nextDate = newUtcObj.getUTCDate();
    } else if ( hours > 12 || (hours === 12 && minutes >= 1) ) {
      nextHour = 18;
    } else if ( hours > 6 || (hours === 6 && minutes >= 1) ) {
      nextHour = 12;
    } else {
      nextHour = 6;
    }
  
    const nextUtcMilliSeconds = Date.UTC(nextYear, nextMonth, nextDate, nextHour, nextMinute, nextSecond);
    // const nextUtcObj = new Date(nextUtcMilliSeconds);
    // console.log(nextUtcObj);
  
    const remainingMilliSeconds = nextUtcMilliSeconds - utcObj.valueOf();
  
  
    let remainingSeconds = Math.round(remainingMilliSeconds / 1000);
  
    let HH:any = Math.floor(Math.floor(remainingSeconds / 60) / 60);
  
    remainingSeconds = remainingSeconds - (HH * 60 * 60);
  
    let MM:any = Math.floor(remainingSeconds / 60);
    let SS:any = remainingSeconds - (MM * 60);
  
    HH = HH < 10 ? `0${HH}` : HH;
    MM = MM < 10 ? `0${MM}` : MM;
    SS = SS < 10 ? `0${SS}` : SS;
    return `${HH}:${MM}:${SS}`;
  }
  
  const now_date_and_time = new Date();
  
  // const data = nextpredictionUpdate(now_date_and_time);
  // console.log(data);

  //Next Data Update
  setInterval(function(){
    const nowDateandTime = new Date();
    let MM:any = nowDateandTime.getUTCMinutes() < 1 ? 0 : 60 - nowDateandTime.getUTCMinutes();
    let SS:any = nowDateandTime.getUTCSeconds() < 1 ? 0 : 60 - nowDateandTime.getUTCSeconds();
  
    MM = MM < 10 ? `0${MM}` : MM;
    SS = SS < 10 ? `0${SS}` : SS;
  
    // var target = `${HH}:${MM}:${SS} `
    var target = `00:${MM}:${SS}`;
    if(target === "00:00:01"){
      HandleLastRowData()
        getLastDayData()
        getBarChartData()
        getPieChartData()
    }
    setUpdatedTime(target)
  
  }, 1000)


  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-6">
      <StatsCard
        icon={
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
        }
        text="Available Spaces"
      >
        {spaces}
      </StatsCard>
      <StatsCard
        icon={
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"
            />
          </svg>
        }
        text="Rows of Parking Data"
      >
        {total_data_rows ? total_data_rows : "0" }
      </StatsCard>
      <StatsCard
        icon={
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        }
        text="Next Data Update"
      >
        {updatedtime }
      </StatsCard>
      <StatsCard
        icon={
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        }
        text="Next Prediction Update"
      >
        {nextpredictionUpdate(now_date_and_time)}
      </StatsCard>
    </div>
  )
}
export { Stats }
