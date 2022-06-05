import { useSelector } from 'react-redux'
import { AreaChart, XAxis, YAxis, CartesianGrid, Tooltip, Area } from 'recharts'

import { ChartCard } from '../chart/ChartCard'

const data = [
  {
      "time": "3am",
      "A": 1622,
      "B": 1118,
      "C": 1856,
      "D": 1243,
      "H": 1284,
      "I": 1231,
      "Libra": 1001
  },
  {
      "time": "4am",
      "A": 1622,
      "B": 1118,
      "C": 1856,
      "D": 1244,
      "H": 1284,
      "I": 1231,
      "Libra": 1001
  },
  {
      "time": "5am",
      "A": 1621,
      "B": 1118,
      "C": 1857,
      "D": 1244,
      "H": 1284,
      "I": 1231,
      "Libra": 1001
  },
  {
      "time": "6am",
      "A": 1619,
      "B": 1118,
      "C": 1856,
      "D": 1245,
      "H": 1284,
      "I": 1231,
      "Libra": 996
  },
  {
      "time": "7am",
      "A": 1614,
      "B": 1118,
      "C": 1854,
      "D": 1244,
      "H": 1284,
      "I": 1231,
      "Libra": 991
  },
  {
      "time": "8am",
      "A": 1587,
      "B": 1118,
      "C": 1837,
      "D": 1267,
      "H": 1284,
      "I": 1231,
      "Libra": 988
  },
  {
      "time": "9am",
      "A": 1565,
      "B": 1118,
      "C": 1798,
      "D": 1284,
      "H": 1284,
      "I": 1231,
      "Libra": 984
  },
  {
      "time": "10am",
      "A": 1540,
      "B": 1118,
      "C": 1750,
      "D": 1293,
      "H": 1284,
      "I": 1231,
      "Libra": 983
  },
  {
      "time": "11am",
      "A": 1520,
      "B": 1118,
      "C": 1735,
      "D": 1310,
      "H": 1284,
      "I": 1231,
      "Libra": 985
  },
  {
      "time": "12pm",
      "A": 1504,
      "B": 1118,
      "C": 1736,
      "D": 1342,
      "H": 1284,
      "I": 1231,
      "Libra": 989
  },
  {
      "time": "1pm",
      "A": 1486,
      "B": 1118,
      "C": 1752,
      "D": 1377,
      "H": 1284,
      "I": 1231,
      "Libra": 994
  },
  {
      "time": "2pm",
      "A": 1473,
      "B": 1118,
      "C": 1766,
      "D": 1429,
      "H": 1284,
      "I": 1231,
      "Libra": 998
  },
  {
      "time": "3pm",
      "A": 1459,
      "B": 1118,
      "C": 1785,
      "D": 1465,
      "H": 1284,
      "I": 1231,
      "Libra": 1014
  },
  {
      "time": "4pm",
      "A": 1444,
      "B": 1118,
      "C": 1818,
      "D": 1507,
      "H": 1284,
      "I": 1231,
      "Libra": 1031
  },
  {
      "time": "5pm",
      "A": 1441,
      "B": 1118,
      "C": 1841,
      "D": 1530,
      "H": 1284,
      "I": 1231,
      "Libra": 1048
  },
  {
      "time": "6pm",
      "A": 1435,
      "B": 1118,
      "C": 1860,
      "D": 1581,
      "H": 1284,
      "I": 1231,
      "Libra": 1051
  },
  {
      "time": "7pm",
      "A": 1435,
      "B": 1118,
      "C": 1878,
      "D": 1600,
      "H": 1284,
      "I": 1231,
      "Libra": 1053
  },
  {
      "time": "8pm",
      "A": 1433,
      "B": 1118,
      "C": 1883,
      "D": 1625,
      "H": 1284,
      "I": 1231,
      "Libra": 1053
  },
  {
      "time": "9pm",
      "A": 1431,
      "B": 1118,
      "C": 1887,
      "D": 1639,
      "H": 1284,
      "I": 1231,
      "Libra": 1054
  },
  {
      "time": "10pm",
      "A": 1425,
      "B": 1118,
      "C": 1901,
      "D": 1649,
      "H": 1284,
      "I": 1231,
      "Libra": 1054
  },
  {
      "time": "11pm",
      "A": 1425,
      "B": 1118,
      "C": 1904,
      "D": 1654,
      "H": 1284,
      "I": 1231,
      "Libra": 1058
  },
  {
      "time": "12am",
      "A": 1423,
      "B": 1118,
      "C": 1912,
      "D": 1656,
      "H": 1284,
      "I": 1231,
      "Libra": 1060
  },
  {
      "time": "1am",
      "A": 1623,
      "B": 1118,
      "C": 1855,
      "D": 1242,
      "H": 1284,
      "I": 1231,
      "Libra": 1061
  },
  {
      "time": "2am",
      "A": 1622,
      "B": 1118,
      "C": 1854,
      "D": 1243,
      "H": 1284,
      "I": 1231,
      "Libra": 1061
  }
]

const LineGraph = () => {
  const state = useSelector((state: any) => state.chart);
  return (
    <ChartCard title="Parking Data">
      <AreaChart
        data={ state.lineChartData.length === 0 ? data : state.lineChartData }
        margin={{
          top: 0,
          right: 30,
          left: 0,
          bottom: 0
        }}
      >
        <XAxis dataKey="time" tickLine={false} axisLine={false} />
        <YAxis
          tickLine={false}
          axisLine={false}
          ticks={state.lineChartTicks.length === 0 ? [0, 500, 1000, 1500, 2000] : state.lineChartTicks}
          tickCount={state.lineChartTicks.length}
          dy={-5}
          dx={-5}
        />
        <CartesianGrid className="dark:stroke-slate-600 stroke-slate-300" strokeDasharray="15" vertical={false} />
        <Tooltip />
        <Area
          type="monotone"
          dataKey="A"
          name="Garage A Available Spaces"
          strokeWidth={5}
          stroke="#3694da"
          fillOpacity={0}
        />
        <Area
          type="monotone"
          dataKey="B"
          name="Garage B Available Spaces"
          strokeWidth={5}
          stroke="#0d9252"
          fillOpacity={0}
        />
        <Area
          type="monotone"
          dataKey="C"
          name="Garage C Available Spaces"
          strokeWidth={5}
          stroke="#556d8c"
          fillOpacity={0}
        />
        <Area
          type="monotone"
          dataKey="D"
          name="Garage D Available Spaces"
          strokeWidth={5}
          stroke="#674ea7"
          fillOpacity={0}
        />
        <Area
          type="monotone"
          dataKey="H"
          name="Garage H Available Spaces"
          strokeWidth={5}
          stroke="#dc1010"
          fillOpacity={0}
        />
        <Area
          type="monotone"
          dataKey="I"
          name="Garage I Available Spaces"
          strokeWidth={5}
          stroke="#b7950c"
          fillOpacity={0}
        />
        <Area
          type="monotone"
          dataKey="Libra"
          name="Garage Libra Available Spaces"
          strokeWidth={5}
          stroke="#bd26ae"
          fillOpacity={0}
        />
      </AreaChart>
    </ChartCard>
  )
}

export { LineGraph }
