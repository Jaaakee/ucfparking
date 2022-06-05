import { BarGraph } from './BarGraph'
import { LineGraph } from './LineGraph'
import { PieGraph } from './PieGraph'

const Charts = () => {
  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12">
        <LineGraph />
      </div>
      <div className="col-span-12 lg:col-span-8">
        <BarGraph />
      </div>
      <div className="col-span-12 lg:col-span-4">
        <PieGraph />
      </div>
    </div>
  )
}

export { Charts }
