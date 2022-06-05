import moment from 'moment'
import 'moment-timezone'
export const handelChartData = (data: any, type: string) => {
  let chartData: any[] = []
  let timebaseLineData: any = {}
  let maxSpaceLeft: number = 0
  let lineGraphTicks: number[] = []

  data.forEach((chart: any) => {
    const { date_and_time, garages } = chart

    if (type === 'lastday') {
      timebaseLineData = {
        time: moment.utc(date_and_time).tz('America/New_York').format('ha')
      }
    } else {
      timebaseLineData = { time: moment(date_and_time).format('DD/MM') }
    }

    for (let garageName in garages) {
      timebaseLineData[garageName] = garages[garageName].spaces_left
      if(garages[garageName].space_left < maxSpaceLeft) {
      }
      maxSpaceLeft = maxSpaceLeft < garages[garageName].spaces_left ? garages[garageName].spaces_left : maxSpaceLeft;      
    }
    chartData.push(timebaseLineData)
  })

  while (maxSpaceLeft % 500 != 0) {
    maxSpaceLeft++;
  }

  for (let i = 0; i <= maxSpaceLeft; i += 500) {
    lineGraphTicks.push(i)
  }

  chartData.reverse()
  return {chartData, lineGraphTicks}
}

export const handleBarChart = (data: any, type: string) => {
  const barChartData = []
  let garages = {}

  for (const garageName in data) {
    if (type === 'barchart') {
      garages = {
        name: `Garage ${garageName}`,
        spaces: data[garageName].spaces_left
      }
    } else {
      garages = {
        name: `Garage ${garageName}`,
        value: data[garageName].spaces_left
      }
    }

    barChartData.push(garages)
  }

  return barChartData
}

const getUniquArray = (data: any) => {
  const arr = []
  let uniqArr: any = []

  for (let index = 0; index < data.length; index++) {
    const date = moment(data[index].date_and_time).format('MM DD')
    arr.push(date)
    uniqArr = arr.filter(function (x, i, a) {
      return a.indexOf(x) === i
    })
  }

  return uniqArr
}

const sumArray = (data: any) => {
  let weeklyData: any = {}
  const chartData: any[] = []
  let maxSpaceLeft: number = 0
  let lineGraphTicks: number[] = []

  data.forEach((item: any) => { 
    const { date, garages } = item
    weeklyData = { time: `${moment(date).format('MM/DD')}` }  
    for (const garage in garages) {
      for (const grageName in garages[garage]) {
        weeklyData[grageName] = garages[garage][grageName].spaces_left
      }
    }

    chartData.push(weeklyData)
  })
  
  for (let i =0; i < chartData.length; i++) {

    if (chartData[i].A > maxSpaceLeft){
      maxSpaceLeft = chartData[i].A
    }

    if (chartData[i].B > maxSpaceLeft){
      maxSpaceLeft = chartData[i].B
    }

    if (chartData[i].C > maxSpaceLeft){
      maxSpaceLeft = chartData[i].C
    }

    if (chartData[i].D > maxSpaceLeft){
      maxSpaceLeft = chartData[i].D
    }

    if (chartData[i].H > maxSpaceLeft){
      maxSpaceLeft = chartData[i].H
    }

    if (chartData[i].I > maxSpaceLeft){
      maxSpaceLeft = chartData[i].I
    }

    if (chartData[i].Libra > maxSpaceLeft){
      maxSpaceLeft = chartData[i].Libra
    }
  }

  while (maxSpaceLeft % 500 != 0) {
    maxSpaceLeft++;
  }

  for (let i = 0; i <= maxSpaceLeft; i += 500) {
    lineGraphTicks.push(i)
  }

  return { chartData, lineGraphTicks }
}

export const filterByWeek = (data: any) => {
  const chartData = []
  let obj = {}
  const days = getUniquArray(data)

  for (let i = 0; i < days.length; i++) {
    const res = data.filter(
      (item: any) => moment(item.date_and_time).format('MM DD') === days[i]
    )

    const garages = res.map((item: any) => item.garages)
    
    obj = { date: res[0].date_and_time, garages: { ...garages } }
    chartData.push(obj)
  }
  
  const space_left = sumArray(chartData)
  space_left.chartData.reverse()
  return space_left
}
